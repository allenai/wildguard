from .wildguard import WildGuard, WildGuardVLLM, WildGuardHF

def load_wildguard(
        use_vllm: bool = True,
        device: str = 'cuda',
        ephemeral_model: bool = True,
) -> WildGuard:
    """
    Loads a WildGuard model for classification.
    @param use_vllm: Whether to use a VLLM model for classification. If False, uses a HuggingFace model.
                     Using VLLM is recommended for better performance.
    @param device: The device to run the HuggingFace model on. Ignored if using VLLM. Default: 'cuda'.
    @param ephemeral_model: Whether to remove the model from the device and free GPU memory after calling classify().
                            Set this to False if you will be calling classify() multiple times. Default: True.
    """
    if use_vllm:
        return WildGuardVLLM(ephemeral_model=ephemeral_model)
    else:
        return WildGuardHF(device=device, ephemeral_model=ephemeral_model)
