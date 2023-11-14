from __future__ import annotations
import configparser
from src.utils import get_tag_path

class Params:
    """
    This class holds the parameters defined in params.ini

    It's basically a container to make it more "object-oriented"
    and (hopefully) easier to use.
    
    For example having a config file:
    ```
    # params.ini

    [Params]
    learning_rate = 0.01
    epochs = 3000
    ```

    We have::
        >>> params = Params()
        >>> params.learning_rate
        0.01
        >>> params.epochs
        3000

    You can also do this:
        >>> params = Params(learning_rate=0.05)
        >>> params.learning_rate
        0.05
        >>> params.epochs
        3000
    """
    def __init__(self, filename="params.ini", **kwargs):
        config = configparser.ConfigParser()
        config.read(filename)
        self.epochs                 = self._getint("epochs", config, **kwargs)
        self.layers                 = self._getint("layers", config, **kwargs)
        self.neurons_per_layer      = self._getint("neurons_per_layer", config, **kwargs)
        self.learning_rate          = self._getfloat("learning_rate", config, **kwargs)

        self.equation               = self._getstr("equation", config, **kwargs)
        self.epsilon                = self._getfloat("epsilon", config, **kwargs)
        self.n_points_x             = self._getint("n_points_x", config, **kwargs)
        self.n_points_t             = self._getint("n_points_t", config, **kwargs)
        self.n_points_x_error       = self._getint("n_points_x_error", config, **kwargs)
        self.n_points_t_error       = self._getint("n_points_t_error", config, **kwargs)
        self.n_test_x               = self._getint("n_test_x", config, **kwargs)
        self.n_test_t               = self._getint("n_test_t", config, **kwargs)
        self.integration_rule_loss  = self._getstr("integration_rule_loss", config, **kwargs)
        self.integration_rule_error = self._getstr("integration_rule_error", config, **kwargs)
        self.tag                    = self._getstr("tag", config, **kwargs)

    def _getstr(self, name: str, config: configparser.ConfigParser, **kwargs) -> str:
        config_value = config["Params"].get(name)
        return kwargs.get(name, config_value)

    def _getfloat(self, name: str, config: configparser.ConfigParser, **kwargs) -> float:
        config_value = config["Params"].getfloat(name)
        return kwargs.get(name, config_value)
    
    def _getint(self, name: str, config: configparser.ConfigParser, **kwargs) -> int:
        config_value = config["Params"].getint(name)
        return kwargs.get(name, config_value)

    def _getboolean(self, name: str, config: configparser.ConfigParser, **kwargs) -> bool:
        config_value = config["Params"].getboolean(name)
        return kwargs.get(name, config_value)

    def save(self, filename: str):
        params = self.__dict__
        config = configparser.ConfigParser()
        config["Params"] = params
        with open(filename, 'w') as f:
            config.write(f)

    def save_by_tag(self):
        tag = self.tag
        filename = f"{get_tag_path(tag)}/params.ini"
        self.save(filename)

    @classmethod
    def load_by_tag(cls, tag, **kwargs) -> Params:
        filename = f"{get_tag_path(tag)}/params.ini"
        return cls(filename=filename, **kwargs)
