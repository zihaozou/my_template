import time
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, List, MutableMapping, MutableSequence, Union
import hydra
from hydra.utils import get_class, get_method, get_object, get_original_cwd
from hydra.experimental.callbacks import Callback
from omegaconf._utils import is_primitive_type_annotation
from omegaconf import DictConfig, OmegaConf, ListConfig, open_dict
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning import Callback
import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path
from src.utils import pylogger, rich_utils
import git
import prettytable

OmegaConf.register_new_resolver("_get_class_", get_class)
OmegaConf.register_new_resolver("_get_method_", get_method)
OmegaConf.register_new_resolver("_get_object_", get_object)

log = pylogger.get_pylogger(__name__)

DEBUG_STATE = False


def set_debug_state(state: bool) -> None:
    global DEBUG_STATE
    DEBUG_STATE = state


def no_debug(fn):
    """Decorator that skips the function if debug is set to True."""

    def wrapper(*args, **kwargs):
        if DEBUG_STATE:
            log.debug(f"Skipping {fn.__name__} because debug is set to True!")
            return None
        return fn(*args, **kwargs)

    return wrapper


@no_debug
@rank_zero_only
def check_git_version(cfg) -> None:
    """Checks if the git version is correct."""
    # save git version to current folder
    current_git_version = git.Repo(get_original_cwd()).head.object.hexsha
    cfg_git_version = cfg.get("git_version")
    if not cfg_git_version:
        log.info("Git version not found in config, skipping git version check...")
    else:
        log.info(
            "Git version found in config, checking if it matches the current git version..."
        )
        if cfg_git_version != current_git_version:
            raise Exception(
                f"Git version mismatch! <cfg.git_version={cfg_git_version}>\n<current_git_version={current_git_version}>\n"
                "Please checkout the correct git version!"
            )
    output_path = HydraConfig.get().run.dir
    OmegaConf.load(Path(output_path) / ".hydra" / "config.yaml")
    with open_dict(cfg):
        cfg.git_version = current_git_version
    OmegaConf.save(cfg, Path(output_path) / ".hydra" / "config.yaml")
    log.info("Git version check passed!")


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):
        set_debug_state(cfg.get("task_name") == "debug")
        # apply extra utilities
        extras(cfg)
        # # convert config first
        # log.info("Converting config...")
        # cfg = convert_config(cfg)
        # check git version
        check_git_version(cfg)
        # execute the task
        try:
            start_time = time.time()
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = (
                f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            )
            save_file(
                path, content
            )  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_plugins(plugins_cfg: DictConfig):
    """Instantiates callbacks from config."""
    plugins = []

    if not plugins_cfg:
        log.warning("Plugins config is empty.")
        return plugins

    if not isinstance(plugins_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, pg_conf in plugins_cfg.items():
        if isinstance(pg_conf, DictConfig) and "_target_" in pg_conf:
            log.info(f"Instantiating callback <{pg_conf._target_}>")
            plugins.append(hydra.utils.instantiate(pg_conf))

    return plugins


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def convert_config(
    config: Union[MutableMapping, MutableSequence]
) -> Union[MutableMapping, MutableSequence]:
    """
    helper function to convert every class, method, object in the config to the actual class, method, object
    """
    if isinstance(config, MutableMapping):
        keys = list(config.keys())
        if len(keys) == 1:
            key = keys[0]
            if key == "_get_class_":
                class_path = config[key]
                if not isinstance(class_path, str):
                    raise TypeError("class path must be a string!")
                class_ = get_class(config[key])
                return class_
            elif key == "_get_method_":
                method_path = config[key]
                if not isinstance(method_path, str):
                    raise TypeError("method path must be a string!")
                method_ = get_method(config[key])
                return method_
            elif key == "_get_object_":
                object_path = config[key]
                if not isinstance(object_path, str):
                    raise TypeError("object path must be a string!")
                object_ = get_object(config[key])
                return object_
        for key in keys:
            if isinstance(config[key], (MutableMapping, MutableSequence)):
                ret_obj = convert_config(config[key])
                if not is_primitive_type_annotation(ret_obj) and isinstance(
                    config, DictConfig
                ):
                    config._set_flag("allow_objects", True)
                config[key] = ret_obj
    elif isinstance(config, MutableSequence):
        for idx in range(len(config)):
            if isinstance(config[idx], (MutableMapping, MutableSequence)):
                ret_obj = convert_config(config[idx])
                if not is_primitive_type_annotation(ret_obj) and isinstance(
                    config, ListConfig
                ):
                    config._set_flag("allow_objects", True)
                config[idx] = ret_obj
    return config


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # get current working directory
    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@no_debug
@rank_zero_only
def zip_source(root_dir, log_dir, task_name):
    """Zips source code for reproducibility."""
    log.info("Checking the Repository and zipping source code...")
    repo = git.Repo(root_dir)
    if repo.is_dirty():
        unstaged_files = repo.index.diff(None)
        changed_file_table = prettytable.PrettyTable()
        changed_file_table.field_names = [
            "File Path",
            "New File?",
            "New Path",
            "Change Type",
        ]
        for file in unstaged_files:
            changed_file_table.add_row(
                [file.a_path, file.new_file, file.b_path, file.change_type]
            )
        log.error("Changed Files:")
        log.error("\n" + str(changed_file_table))
        raise Exception(
            "Repository is dirty! Please commit all changes before running the experiment. If you are debugging and don't want to commit changes, please set 'debug=default' in the config file."
        )
    log.info("Repository is clean, creating tag...")
    repo.create_tag(
        task_name, message=f"Tagging the commit for {task_name}", force=True
    )

    with open(Path(log_dir) / "source_code.zip", "wb") as f:
        log.info("Zipping source code...")
        repo.archive(f, format="zip")

    log.info("Zipping source code completed.")

    return
