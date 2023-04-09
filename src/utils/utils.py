import time
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path
from src.utils import pylogger, rich_utils
import git
import prettytable
log = pylogger.get_pylogger(__name__)


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

        # apply extra utilities
        extras(cfg)

        # execute the task
        try:
            start_time = time.time()
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            save_file(path, content)  # save task execution time (even if exception occurs)
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

def zip_source(root_dir,log_dir,task_name):
    """Zips source code for reproducibility."""
    log.info("Checking the Repository and zipping source code...")
    repo = git.Repo(root_dir)
    if repo.is_dirty():
        unstaged_files = repo.index.diff(None)
        changed_file_table = prettytable.PrettyTable()
        changed_file_table.field_names = ["File Path", "New File?", "New Path", "Change Type"]
        for file in unstaged_files:
            changed_file_table.add_row([file.a_path, file.new_file, file.b_path, file.change_type])
        log.error("Changed Files:")
        log.error('\n'+str(changed_file_table))
        raise Exception("Repository is dirty! Please commit all changes before running the experiment. If you are debugging and don't want to commit changes, please set 'debug=default' in the config file.")
    log.info("Repository is clean, creating tag...")
    repo.create_tag(task_name, message=f"Tagging the commit for {task_name}", force=True)

    with open(Path(log_dir) / "source_code.zip", "wb") as f:
        log.info("Zipping source code...")
        repo.archive(f, format="zip")

    log.info("Zipping source code completed.")

    return