from .wildguard import WildGuard, WildGuardVLLM, WildGuardHF

def load_wildguard(
        use_vllm: bool = True,
        device: str = 'cuda',
        ephemeral_model: bool = True,
        batch_size: int | None = None,
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
        if batch_size == None:
            batch_size = -1
        return WildGuardVLLM(ephemeral_model=ephemeral_model, batch_size=batch_size)
    else:
        if batch_size == None:
            batch_size = 16
        return WildGuardHF(device=device, ephemeral_model=ephemeral_model, batch_size=batch_size)
