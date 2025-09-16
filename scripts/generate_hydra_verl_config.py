import verl
import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="config/helper", config_name="verl_hydra_gen", version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()
