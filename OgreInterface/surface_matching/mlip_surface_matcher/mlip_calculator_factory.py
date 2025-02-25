import typing as tp

from ase.calculators.mixing import SumCalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator


def get_combined_calculator(mlip: str, model_name: tp.Optional[str], device: str):
    mlip_upper = mlip.upper()

    # Initialize DFT-D dispersion calculator
    dftd_calc = TorchDFTD3Calculator(
        device=device,
        old=True,
        xc="pbe"
    )

    # Initialize selected MLIP model
    mlip_calc = None
    if mlip_upper == "CHGNET":
        try:
            from chgnet.model import CHGNetCalculator
            mlip_calc = CHGNetCalculator(
                use_device=device
            )
        except ImportError:
            raise ImportError("You need to install `chgnet` to use CHGNet calculator.")
    elif mlip_upper == "MACE":
        if model_name is None:
            raise ValueError("MACE model name must be provided when using MACE.")
        try:
            from mace.calculators import mace_mp
            mlip_calc = mace_mp(
                model=f"https://github.com/ACEsuit/mace-mp/releases/download/{model_name}.model",
                dispersion=False,
                default_dtype="float64",
                device=device,
            )
        except ImportError:
            raise ImportError("You need to install `mace-torch` to use MACE-MP.")
    elif mlip_upper == "DP":
        if model_name is None:
            raise ValueError("DeepMD model path must be provided when using DeepMD.")
        try:
            from deepmd.calculator import DP
            mlip_calc = DP(model_name)
        except ImportError:
            raise ImportError("You need to install `deepmd-kit` to use DeepMD.")
    else:
        raise ValueError("Invalid MLIP selection. Choose from: `CHGNet`, `MACE`, `DP`.")

    # Combine MLIP and DFT-D calculators
    return SumCalculator([mlip_calc, dftd_calc])
