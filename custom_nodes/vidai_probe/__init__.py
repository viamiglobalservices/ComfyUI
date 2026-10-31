import torch


class LatentToImageProbe:
    """Save the raw sampler latent plus its statistics, to diagnose divergence."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "probe"
    CATEGORY = "vidai/debug"

    def probe(self, samples):
        t = samples["samples"] if isinstance(samples, dict) else samples
        flat = t.detach().to(torch.float32)
        stats = (f"latent shape={tuple(t.shape)} min={flat.min():.4f} max={flat.max():.4f} "
                 f"mean={flat.mean():.4f} std={flat.std():.4f} "
                 f"absmax={flat.abs().max():.4f} nan={bool(torch.isnan(flat).any())} "
                 f"inf={bool(torch.isinf(flat).any())}")
        print("[vidai-latent-probe]", stats, flush=True)
        with open("/tmp/latent_stats.txt", "a") as fh:
            fh.write(stats + "\n")

        # Convert latent -> image using ComfyUI's own VAE-free normalisation
        # so the result is viewable: scale SDXL latents by 1/0.13025.
        scaled = flat / 0.13025
        img = torch.clamp(scaled * 0.5 + 0.5, 0, 1)
        img = img.movedim(1, -1).repeat(1, 1, 1, 3)
        return (img,)


NODE_CLASS_MAPPINGS = {"LatentToImageProbe": LatentToImageProbe}
NODE_DISPLAY_NAME_MAPPINGS = {"LatentToImageProbe": "vidAI Latent Probe"}
