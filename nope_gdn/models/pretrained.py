import torch


def load_pretrained_vit(model, checkpoint_name="deit_small_patch16_224", verbose=True):
    """
    Initialize NoPE+GDN encoder from a pretrained 2D ViT / DeiT.

    Transfers encoder blocks (attention, MLP, norms) and inflates the
    2D patch embedding into a 3D tubelet embedding. Skips positional
    embeddings (NoPE doesn't use them), GDN layers, and classifier head.

    Key mapping (DeiT -> NoPE+GDN):
        patch_embed.proj          -> backbone.encoder.tubelet_embed.projection (inflated)
        blocks.{i}.norm1          -> backbone.encoder.blocks.{i}.norm1
        blocks.{i}.attn.qkv      -> backbone.encoder.blocks.{i}.attn.qkv_proj
        blocks.{i}.attn.proj     -> backbone.encoder.blocks.{i}.attn.out_proj
        blocks.{i}.norm2          -> backbone.encoder.blocks.{i}.norm2
        blocks.{i}.mlp.fc1       -> backbone.encoder.blocks.{i}.mlp.0
        blocks.{i}.mlp.fc2       -> backbone.encoder.blocks.{i}.mlp.3
        norm                      -> backbone.encoder.norm
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm required: pip install timm")

    vit = timm.create_model(checkpoint_name, pretrained=True)
    vit_sd = vit.state_dict()
    model_sd = model.state_dict()

    loaded = []
    skipped = []

    # ---- 1. Tubelet embed: inflate 2D patch_embed -> 3D Conv3d ----
    tubelet_key = "backbone.encoder.tubelet_embed.projection.weight"
    patch_key = "patch_embed.proj.weight"

    if patch_key in vit_sd and tubelet_key in model_sd:
        vit_w = vit_sd[patch_key]                     # (D, C, H, W)
        T_tube = model_sd[tubelet_key].shape[2]       # temporal tubelet size
        inflated = vit_w.unsqueeze(2).repeat(1, 1, T_tube, 1, 1) / T_tube
        model_sd[tubelet_key] = inflated
        loaded.append(f"  tubelet_embed.projection.weight (inflated 2D->3D, T={T_tube})")

    tubelet_bias_key = "backbone.encoder.tubelet_embed.projection.bias"
    patch_bias_key = "patch_embed.proj.bias"
    if patch_bias_key in vit_sd and tubelet_bias_key in model_sd:
        if vit_sd[patch_bias_key].shape == model_sd[tubelet_bias_key].shape:
            model_sd[tubelet_bias_key] = vit_sd[patch_bias_key]
            loaded.append("  tubelet_embed.projection.bias")

    # ---- 2. Encoder blocks ----
    encoder_block_count = sum(
        1 for k in model_sd if k.startswith("backbone.encoder.blocks.") and k.endswith(".norm1.weight"))
    vit_block_count = sum(
        1 for k in vit_sd if k.startswith("blocks.") and k.endswith(".norm1.weight"))
    num_transfer = min(encoder_block_count, vit_block_count)

    for i in range(num_transfer):
        block_maps = [
            ("norm1.weight",      "norm1.weight"),
            ("norm1.bias",        "norm1.bias"),
            ("attn.qkv.weight",   "attn.qkv_proj.weight"),
            ("attn.qkv.bias",     "attn.qkv_proj.bias"),
            ("attn.proj.weight",  "attn.out_proj.weight"),
            ("attn.proj.bias",    "attn.out_proj.bias"),
            ("norm2.weight",      "norm2.weight"),
            ("norm2.bias",        "norm2.bias"),
            ("mlp.fc1.weight",    "mlp.0.weight"),
            ("mlp.fc1.bias",      "mlp.0.bias"),
            ("mlp.fc2.weight",    "mlp.3.weight"),
            ("mlp.fc2.bias",      "mlp.3.bias"),
        ]

        for vit_suffix, model_suffix in block_maps:
            vit_key = f"blocks.{i}.{vit_suffix}"
            model_key = f"backbone.encoder.blocks.{i}.{model_suffix}"

            if vit_key not in vit_sd:
                continue
            if model_key not in model_sd:
                skipped.append(f"  block {i}: {vit_suffix} (no target)")
                continue
            if vit_sd[vit_key].shape != model_sd[model_key].shape:
                skipped.append(f"  block {i}: {vit_suffix} shape mismatch")
                continue

            model_sd[model_key] = vit_sd[vit_key]
            loaded.append(f"  encoder.blocks.{i}.{model_suffix}")

    # ---- 3. Encoder final norm ----
    for suffix in ["weight", "bias"]:
        vit_key = f"norm.{suffix}"
        model_key = f"backbone.encoder.norm.{suffix}"
        if vit_key in vit_sd and model_key in model_sd:
            if vit_sd[vit_key].shape == model_sd[model_key].shape:
                model_sd[model_key] = vit_sd[vit_key]
                loaded.append(f"  encoder.norm.{suffix}")

    # ---- 4. Load ----
    model.load_state_dict(model_sd)

    if verbose:
        print(f"=" * 60)
        print(f"Pretrained: {checkpoint_name} -> NoPE+GDN")
        print(f"=" * 60)
        print(f"  DeiT/ViT blocks: {vit_block_count}")
        print(f"  Encoder blocks:  {encoder_block_count}")
        print(f"  Transferred:     {num_transfer} blocks")
        print(f"  Loaded: {len(loaded)} parameter tensors")
        if skipped:
            print(f"  Skipped: {len(skipped)}")
            for s in skipped[:5]:
                print(f"    {s}")
        print(f"  Random init: GDN layers, processor blocks, classification head")
        print(f"=" * 60)

    return model


PRETRAINED_MAP = {
    "tiny":  "deit_tiny_patch16_224",
    "small": "deit_small_patch16_224",
    "base":  "vit_base_patch16_224",
}


def load_videomae_pretrained(model, checkpoint_name="MCG-NJU/videomae-base", verbose=True):
    """
    Initialize NoPE+GDN encoder from a pretrained VideoMAE model.

    VideoMAE is self-supervised (masked video reconstruction) so the encoder
    already understands temporal structure -- much better than DeiT for SSv2.

    Key differences from DeiT loading:
      - Tubelet embed is ALREADY 3D Conv3d (no inflation needed)
      - Q, K, V are SEPARATE matrices (must concat into qkv_proj)
      - Positional embeddings are skipped (NoPE doesn't use them)
      - Decoder weights are skipped (we only want the encoder)

    Key mapping (VideoMAE HF -> NoPE+GDN):
        videomae.embeddings.patch_embeddings.projection  -> backbone.encoder.tubelet_embed.projection
        videomae.encoder.layer.{i}.layernorm_before       -> backbone.encoder.blocks.{i}.norm1
        videomae.encoder.layer.{i}.attention.attention.query/key/value -> backbone.encoder.blocks.{i}.attn.qkv_proj (concatenated)
        videomae.encoder.layer.{i}.attention.output.dense  -> backbone.encoder.blocks.{i}.attn.out_proj
        videomae.encoder.layer.{i}.layernorm_after         -> backbone.encoder.blocks.{i}.norm2
        videomae.encoder.layer.{i}.intermediate.dense      -> backbone.encoder.blocks.{i}.mlp.0
        videomae.encoder.layer.{i}.output.dense            -> backbone.encoder.blocks.{i}.mlp.3
        videomae.layernorm                                 -> backbone.encoder.norm
    """
    try:
        from transformers import VideoMAEForPreTraining
    except ImportError:
        raise ImportError("transformers required: pip install transformers")

    print(f"Downloading VideoMAE: {checkpoint_name} ...")
    vmae = VideoMAEForPreTraining.from_pretrained(checkpoint_name)
    vmae_sd = vmae.state_dict()
    model_sd = model.state_dict()

    loaded = []
    skipped = []

    # ---- 1. Tubelet embed (already 3D in VideoMAE) ----
    tube_map = {
        "videomae.embeddings.patch_embeddings.projection.weight":
            "backbone.encoder.tubelet_embed.projection.weight",
        "videomae.embeddings.patch_embeddings.projection.bias":
            "backbone.encoder.tubelet_embed.projection.bias",
    }
    for vmae_key, model_key in tube_map.items():
        if vmae_key in vmae_sd and model_key in model_sd:
            if vmae_sd[vmae_key].shape == model_sd[model_key].shape:
                model_sd[model_key] = vmae_sd[vmae_key]
                loaded.append(f"  tubelet_embed: {model_key.split('.')[-1]}")
            else:
                skipped.append(f"  tubelet_embed: shape mismatch "
                               f"{vmae_sd[vmae_key].shape} vs {model_sd[model_key].shape}")

    # ---- 2. Encoder blocks ----
    encoder_block_count = sum(
        1 for k in model_sd if k.startswith("backbone.encoder.blocks.") and k.endswith(".norm1.weight"))
    vmae_block_count = sum(
        1 for k in vmae_sd if k.startswith("videomae.encoder.layer.") and k.endswith(".layernorm_before.weight"))
    num_transfer = min(encoder_block_count, vmae_block_count)

    for i in range(num_transfer):
        # -- Norms --
        norm_maps = [
            (f"videomae.encoder.layer.{i}.layernorm_before.weight",
             f"backbone.encoder.blocks.{i}.norm1.weight"),
            (f"videomae.encoder.layer.{i}.layernorm_before.bias",
             f"backbone.encoder.blocks.{i}.norm1.bias"),
            (f"videomae.encoder.layer.{i}.layernorm_after.weight",
             f"backbone.encoder.blocks.{i}.norm2.weight"),
            (f"videomae.encoder.layer.{i}.layernorm_after.bias",
             f"backbone.encoder.blocks.{i}.norm2.bias"),
        ]
        for vmae_key, model_key in norm_maps:
            if vmae_key in vmae_sd and model_key in model_sd:
                model_sd[model_key] = vmae_sd[vmae_key]
                loaded.append(f"  block {i}: {model_key.split('.')[-2]}.{model_key.split('.')[-1]}")

        # -- Attention: concat separate Q, K, V -> qkv_proj --
        q_w = vmae_sd.get(f"videomae.encoder.layer.{i}.attention.attention.query.weight")
        k_w = vmae_sd.get(f"videomae.encoder.layer.{i}.attention.attention.key.weight")
        v_w = vmae_sd.get(f"videomae.encoder.layer.{i}.attention.attention.value.weight")
        qkv_key = f"backbone.encoder.blocks.{i}.attn.qkv_proj.weight"

        if q_w is not None and k_w is not None and v_w is not None and qkv_key in model_sd:
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
            if qkv_w.shape == model_sd[qkv_key].shape:
                model_sd[qkv_key] = qkv_w
                loaded.append(f"  block {i}: attn.qkv_proj.weight (Q+K+V concat)")
            else:
                skipped.append(f"  block {i}: qkv shape mismatch {qkv_w.shape} vs {model_sd[qkv_key].shape}")

        q_b = vmae_sd.get(f"videomae.encoder.layer.{i}.attention.attention.query.bias")
        k_b = vmae_sd.get(f"videomae.encoder.layer.{i}.attention.attention.key.bias")
        v_b = vmae_sd.get(f"videomae.encoder.layer.{i}.attention.attention.value.bias")
        qkv_b_key = f"backbone.encoder.blocks.{i}.attn.qkv_proj.bias"

        if q_b is not None and k_b is not None and v_b is not None and qkv_b_key in model_sd:
            qkv_b = torch.cat([q_b, k_b, v_b], dim=0)
            if qkv_b.shape == model_sd[qkv_b_key].shape:
                model_sd[qkv_b_key] = qkv_b
                loaded.append(f"  block {i}: attn.qkv_proj.bias (Q+K+V concat)")

        # -- Attention output projection --
        attn_out_maps = [
            (f"videomae.encoder.layer.{i}.attention.output.dense.weight",
             f"backbone.encoder.blocks.{i}.attn.out_proj.weight"),
            (f"videomae.encoder.layer.{i}.attention.output.dense.bias",
             f"backbone.encoder.blocks.{i}.attn.out_proj.bias"),
        ]
        for vmae_key, model_key in attn_out_maps:
            if vmae_key in vmae_sd and model_key in model_sd:
                if vmae_sd[vmae_key].shape == model_sd[model_key].shape:
                    model_sd[model_key] = vmae_sd[vmae_key]
                    loaded.append(f"  block {i}: {model_key.split('.')[-2]}.{model_key.split('.')[-1]}")

        # -- MLP --
        mlp_maps = [
            (f"videomae.encoder.layer.{i}.intermediate.dense.weight",
             f"backbone.encoder.blocks.{i}.mlp.0.weight"),
            (f"videomae.encoder.layer.{i}.intermediate.dense.bias",
             f"backbone.encoder.blocks.{i}.mlp.0.bias"),
            (f"videomae.encoder.layer.{i}.output.dense.weight",
             f"backbone.encoder.blocks.{i}.mlp.3.weight"),
            (f"videomae.encoder.layer.{i}.output.dense.bias",
             f"backbone.encoder.blocks.{i}.mlp.3.bias"),
        ]
        for vmae_key, model_key in mlp_maps:
            if vmae_key in vmae_sd and model_key in model_sd:
                if vmae_sd[vmae_key].shape == model_sd[model_key].shape:
                    model_sd[model_key] = vmae_sd[vmae_key]
                    loaded.append(f"  block {i}: mlp.{model_key.split('.')[-2]}.{model_key.split('.')[-1]}")

    # ---- 3. Encoder final norm ----
    for suffix in ["weight", "bias"]:
        vmae_key = f"videomae.layernorm.{suffix}"
        model_key = f"backbone.encoder.norm.{suffix}"
        if vmae_key in vmae_sd and model_key in model_sd:
            if vmae_sd[vmae_key].shape == model_sd[model_key].shape:
                model_sd[model_key] = vmae_sd[vmae_key]
                loaded.append(f"  encoder.norm.{suffix}")

    # ---- 4. Load ----
    model.load_state_dict(model_sd)

    # Free VideoMAE model memory
    del vmae, vmae_sd

    if verbose:
        print(f"=" * 60)
        print(f"VideoMAE Pretrained: {checkpoint_name} -> NoPE+GDN")
        print(f"=" * 60)
        print(f"  VideoMAE blocks:  {vmae_block_count}")
        print(f"  Encoder blocks:   {encoder_block_count}")
        print(f"  Transferred:      {num_transfer} blocks")
        print(f"  Loaded: {len(loaded)} params")
        if skipped:
            print(f"  Skipped: {len(skipped)} params")
            for s in skipped:
                print(f"    {s}")
        print(f"")
        print(f"  NOTE: Tubelet embed is native 3D (no inflation)")
        print(f"  NOTE: Q,K,V concatenated into qkv_proj")
        print(f"  NOTE: Positional embeddings skipped (NoPE)")
        print(f"  NOTE: GDN processor + head = random init")
        print(f"=" * 60)

    return model


VIDEOMAE_MAP = {
    "tiny":  None,  # No VideoMAE-Tiny available
    "small": None,  # No VideoMAE-Small available (use DeiT for small)
    "base":  "MCG-NJU/videomae-base",
}


def load_videomae_pretrained_trecvit(
    model, checkpoint_name="MCG-NJU/videomae-base", verbose=True
):
    """Initialize a TRecViTClassifier from a VideoMAE-base checkpoint.

    Transfers the Conv3d patch embed, the 3D position embedding (averaged
    over T to a 2D spatial pos embed), per-block norms/QKV/MLP, and the final
    LayerNorm. Temporal LRU layers, tokenizer.cls_token, pre_logits and
    cls_head stay random (no VideoMAE analog).

    Args:
        model:           A TRecViTClassifier instance.
        checkpoint_name: HuggingFace model id (must be VideoMAE-base, 768-dim).
        verbose:         Print per-tensor transfer log.

    Returns:
        The same model with weights loaded in-place.
    """
    try:
        from transformers import VideoMAEForPreTraining
    except ImportError:
        raise ImportError("transformers required: pip install transformers")

    # Width guard — VideoMAE-base is 768. Refuse anything else.
    width = model.tokenizer.proj.out_channels
    if width != 768:
        raise ValueError(
            f"load_videomae_pretrained_trecvit requires width=768 "
            f"(VideoMAE-base), got width={width}. "
            f"Use size='base' for the trecvit variant."
        )

    print(f"Downloading VideoMAE: {checkpoint_name} ...")
    vmae = VideoMAEForPreTraining.from_pretrained(checkpoint_name)
    vmae_sd = vmae.state_dict()
    model_sd = model.state_dict()

    loaded, skipped = [], []

    # ---- 1. Tokenizer Conv3d ----
    for vk, mk in [
        ("videomae.embeddings.patch_embeddings.projection.weight", "tokenizer.proj.weight"),
        ("videomae.embeddings.patch_embeddings.projection.bias",   "tokenizer.proj.bias"),
    ]:
        if vk in vmae_sd and mk in model_sd:
            if vmae_sd[vk].shape == model_sd[mk].shape:
                model_sd[mk] = vmae_sd[vk]
                loaded.append(f"  {mk}")
            else:
                skipped.append(f"  {mk}: shape mismatch "
                               f"{tuple(vmae_sd[vk].shape)} vs {tuple(model_sd[mk].shape)}")

    # ---- 2. Position embedding: VideoMAE 3D [1, T*N, D] -> mean over T -> [1, N, D] ----
    vk_pos = "videomae.embeddings.position_embeddings"
    mk_pos = "tokenizer.pos_embed"
    if vk_pos in vmae_sd and mk_pos in model_sd:
        v_pos = vmae_sd[vk_pos]                 # [1, T*N, D]
        m_pos = model_sd[mk_pos]                 # [1, N, D]
        n_target = m_pos.shape[1]
        if v_pos.shape[1] % n_target == 0 and v_pos.shape[2] == m_pos.shape[2]:
            T_v = v_pos.shape[1] // n_target
            pooled = v_pos.reshape(1, T_v, n_target, -1).mean(dim=1)  # [1, N, D]
            if pooled.shape == m_pos.shape:
                model_sd[mk_pos] = pooled
                loaded.append(f"  {mk_pos}  (mean over T={T_v})")
            else:
                skipped.append(f"  {mk_pos}: pooled shape {tuple(pooled.shape)} vs target {tuple(m_pos.shape)}")
        else:
            skipped.append(f"  {mk_pos}: incompatible "
                           f"{tuple(v_pos.shape)} vs {tuple(m_pos.shape)}")

    # ---- 3. Spatial encoder blocks (only first min(vmae_depth, trecvit_depth)) ----
    trecvit_depth = sum(
        1 for k in model_sd
        if k.startswith("encoder.spatial.") and k.endswith(".norm1.weight")
    )
    vmae_depth = sum(
        1 for k in vmae_sd
        if k.startswith("videomae.encoder.layer.") and k.endswith(".layernorm_before.weight")
    )
    n_transfer = min(trecvit_depth, vmae_depth)

    for i in range(n_transfer):
        # -- Norms --
        for vk, mk in [
            (f"videomae.encoder.layer.{i}.layernorm_before.weight", f"encoder.spatial.{i}.norm1.weight"),
            (f"videomae.encoder.layer.{i}.layernorm_before.bias",   f"encoder.spatial.{i}.norm1.bias"),
            (f"videomae.encoder.layer.{i}.layernorm_after.weight",  f"encoder.spatial.{i}.norm2.weight"),
            (f"videomae.encoder.layer.{i}.layernorm_after.bias",    f"encoder.spatial.{i}.norm2.bias"),
        ]:
            if vk in vmae_sd and mk in model_sd and vmae_sd[vk].shape == model_sd[mk].shape:
                model_sd[mk] = vmae_sd[vk]
                loaded.append(f"  {mk}")

        # -- Attention: concat Q, K, V into in_proj_{weight,bias} (nn.MultiheadAttention layout) --
        q_w = vmae_sd.get(f"videomae.encoder.layer.{i}.attention.attention.query.weight")
        k_w = vmae_sd.get(f"videomae.encoder.layer.{i}.attention.attention.key.weight")
        v_w = vmae_sd.get(f"videomae.encoder.layer.{i}.attention.attention.value.weight")
        in_w_key = f"encoder.spatial.{i}.attn.in_proj_weight"
        if all(t is not None for t in (q_w, k_w, v_w)) and in_w_key in model_sd:
            qkv = torch.cat([q_w, k_w, v_w], dim=0)
            if qkv.shape == model_sd[in_w_key].shape:
                model_sd[in_w_key] = qkv
                loaded.append(f"  {in_w_key}  (Q+K+V concat)")
            else:
                skipped.append(f"  {in_w_key}: concat {tuple(qkv.shape)} vs {tuple(model_sd[in_w_key].shape)}")

        q_b = vmae_sd.get(f"videomae.encoder.layer.{i}.attention.attention.query.bias")
        k_b = vmae_sd.get(f"videomae.encoder.layer.{i}.attention.attention.key.bias")
        v_b = vmae_sd.get(f"videomae.encoder.layer.{i}.attention.attention.value.bias")
        in_b_key = f"encoder.spatial.{i}.attn.in_proj_bias"
        if all(t is not None for t in (q_b, k_b, v_b)) and in_b_key in model_sd:
            qkv_b = torch.cat([q_b, k_b, v_b], dim=0)
            if qkv_b.shape == model_sd[in_b_key].shape:
                model_sd[in_b_key] = qkv_b
                loaded.append(f"  {in_b_key}  (Q+K+V concat)")

        # -- Attention output projection --
        for vk, mk in [
            (f"videomae.encoder.layer.{i}.attention.output.dense.weight", f"encoder.spatial.{i}.attn.out_proj.weight"),
            (f"videomae.encoder.layer.{i}.attention.output.dense.bias",   f"encoder.spatial.{i}.attn.out_proj.bias"),
        ]:
            if vk in vmae_sd and mk in model_sd and vmae_sd[vk].shape == model_sd[mk].shape:
                model_sd[mk] = vmae_sd[vk]
                loaded.append(f"  {mk}")

        # -- MLP --
        for vk, mk in [
            (f"videomae.encoder.layer.{i}.intermediate.dense.weight", f"encoder.spatial.{i}.mlp.0.weight"),
            (f"videomae.encoder.layer.{i}.intermediate.dense.bias",   f"encoder.spatial.{i}.mlp.0.bias"),
            (f"videomae.encoder.layer.{i}.output.dense.weight",       f"encoder.spatial.{i}.mlp.3.weight"),
            (f"videomae.encoder.layer.{i}.output.dense.bias",         f"encoder.spatial.{i}.mlp.3.bias"),
        ]:
            if vk in vmae_sd and mk in model_sd and vmae_sd[vk].shape == model_sd[mk].shape:
                model_sd[mk] = vmae_sd[vk]
                loaded.append(f"  {mk}")

    # ---- 4. Final LayerNorm ----
    for suffix in ["weight", "bias"]:
        vk = f"videomae.layernorm.{suffix}"
        mk = f"encoder.final_norm.{suffix}"
        if vk in vmae_sd and mk in model_sd and vmae_sd[vk].shape == model_sd[mk].shape:
            model_sd[mk] = vmae_sd[vk]
            loaded.append(f"  {mk}")

    # ---- 5. Apply ----
    model.load_state_dict(model_sd)
    del vmae, vmae_sd

    if verbose:
        n_total = sum(p.numel() for p in model.parameters())
        n_loaded_params = 0
        # Estimate transferred params (re-walk)
        loaded_keys = {ln.strip().split()[0] for ln in loaded}
        for n, p in model.named_parameters():
            if n in loaded_keys:
                n_loaded_params += p.numel()
        # Buffers (pos_embed) — count separately if present
        for k, v in model.state_dict().items():
            if k in loaded_keys and k not in {n for n, _ in model.named_parameters()}:
                n_loaded_params += v.numel()

        print("=" * 60)
        print(f"VideoMAE Pretrained: {checkpoint_name} -> TRecViT")
        print("=" * 60)
        print(f"  VideoMAE blocks:   {vmae_depth}")
        print(f"  TRecViT spatial:   {trecvit_depth}")
        print(f"  Transferred:       {n_transfer} blocks")
        print(f"  Tensors loaded:    {len(loaded)}")
        print(f"  Approx params loaded: {n_loaded_params/1e6:.2f}M / {n_total/1e6:.2f}M "
              f"({100*n_loaded_params/n_total:.1f}%)")
        if skipped:
            print(f"  Skipped:           {len(skipped)}")
            for s in skipped[:5]:
                print(f"    {s}")
        print()
        print("  NOTE: Temporal LRU layers stay RANDOM (no VideoMAE analog)")
        print("  NOTE: tokenizer.cls_token, pre_logits, cls_head also random")
        print("  NOTE: Q,K,V concatenated into nn.MultiheadAttention.in_proj_weight")
        print("=" * 60)

    return model


def load_videomae_into_rope(model,
                            checkpoint_name="MCG-NJU/videomae-base",
                            verbose=True):
    """
    Transfer VideoMAE-Base weights into RoPEVideoClassifier.

    Blocks 0-11: from VideoMAE (Q/K/V with biases, MLPs, LayerNorms)
    Blocks 12-15: random init (no VideoMAE equivalent — RoPE has 16 total)
    Tubelet embedding: direct transfer (same Conv3d)
    VideoMAE pos_embed: skipped (VideoRoPE computes rotary on-the-fly)
    Classification head: skipped (random init)

    Supports both SSL-only and fine-tuned checkpoints by loading via
    VideoMAEModel (the shared bare encoder).
    """
    try:
        from transformers import VideoMAEModel
    except ImportError as e:
        raise ImportError(
            "transformers is required for VideoMAE loading. "
            "Install: pip install transformers") from e

    print(f"Downloading VideoMAE: {checkpoint_name} ...")
    vmae = VideoMAEModel.from_pretrained(checkpoint_name)
    vmae_sd = vmae.state_dict()
    model_sd = model.state_dict()

    loaded = []

    # Tubelet embedding (Conv3d) — VideoMAEModel drops the outer "videomae."
    # prefix that VideoMAEForVideoClassification adds.
    src = "embeddings.patch_embeddings.projection"
    dst = "tubelet_embed.projection"
    for suffix in [".weight", ".bias"]:
        if f"{src}{suffix}" in vmae_sd and f"{dst}{suffix}" in model_sd:
            model_sd[f"{dst}{suffix}"] = vmae_sd[f"{src}{suffix}"]
            loaded.append(f"{dst}{suffix}")

    # Encoder blocks 0-11
    for i in range(12):
        src_pre = f"encoder.layer.{i}"
        dst_pre = f"blocks.{i}"

        mapping = {
            f"{src_pre}.layernorm_before.weight":         f"{dst_pre}.norm1.weight",
            f"{src_pre}.layernorm_before.bias":           f"{dst_pre}.norm1.bias",
            f"{src_pre}.layernorm_after.weight":          f"{dst_pre}.norm2.weight",
            f"{src_pre}.layernorm_after.bias":            f"{dst_pre}.norm2.bias",
            f"{src_pre}.intermediate.dense.weight":       f"{dst_pre}.mlp.0.weight",
            f"{src_pre}.intermediate.dense.bias":         f"{dst_pre}.mlp.0.bias",
            f"{src_pre}.output.dense.weight":             f"{dst_pre}.mlp.3.weight",
            f"{src_pre}.output.dense.bias":               f"{dst_pre}.mlp.3.bias",
            f"{src_pre}.attention.output.dense.weight":   f"{dst_pre}.attn.out_proj.weight",
            f"{src_pre}.attention.output.dense.bias":     f"{dst_pre}.attn.out_proj.bias",
        }
        for s, d in mapping.items():
            if s in vmae_sd and d in model_sd:
                model_sd[d] = vmae_sd[s]
                loaded.append(d)

        # Q/K/V — VideoMAE stores separately, RoPE uses fused qkv_proj
        q_w = vmae_sd.get(f"{src_pre}.attention.attention.query.weight")
        k_w = vmae_sd.get(f"{src_pre}.attention.attention.key.weight")
        v_w = vmae_sd.get(f"{src_pre}.attention.attention.value.weight")
        q_b = vmae_sd.get(f"{src_pre}.attention.attention.query.bias")
        k_b = vmae_sd.get(f"{src_pre}.attention.attention.key.bias")
        v_b = vmae_sd.get(f"{src_pre}.attention.attention.value.bias")

        qkv_w_key = f"{dst_pre}.attn.qkv_proj.weight"
        qkv_b_key = f"{dst_pre}.attn.qkv_proj.bias"

        if qkv_w_key in model_sd and q_w is not None:
            model_sd[qkv_w_key] = torch.cat([q_w, k_w, v_w], dim=0)
            loaded.append(qkv_w_key)
        if qkv_b_key in model_sd and q_b is not None:
            model_sd[qkv_b_key] = torch.cat([q_b, k_b, v_b], dim=0)
            loaded.append(qkv_b_key)

    # Final LayerNorm
    for suffix in [".weight", ".bias"]:
        src_key = f"layernorm{suffix}"
        dst_key = f"norm{suffix}"
        if src_key in vmae_sd and dst_key in model_sd:
            model_sd[dst_key] = vmae_sd[src_key]
            loaded.append(dst_key)

    model.load_state_dict(model_sd)

    # Free VideoMAE memory
    del vmae, vmae_sd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"✅ VideoMAE → RoPE: {len(loaded)} params transferred")
        print(f"   Source: {checkpoint_name}")
        print(f"   Blocks 0-11: from VideoMAE (weights + biases)")
        print(f"   Blocks 12-15: random init (4 extra blocks)")
        print(f"   RoPE: computed on-the-fly (zero learned params)")
        print(f"   pos_embed: skipped (VideoRoPE replaces it)")
        print(f"   head: random init")

    return model
