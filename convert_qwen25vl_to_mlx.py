import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Qwen2.5-VL HF weights to MLX while forcing the slow processor."
    )
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--q-bits", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--skip-vision", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    import mlx_vlm.utils as mlx_utils

    original_load_processor = mlx_utils.load_processor

    def load_processor_with_slow_qwen_processor(
        model_path,
        add_detokenizer=True,
        **kwargs,
    ):
        kwargs.setdefault("use_fast", False)
        return original_load_processor(
            model_path,
            add_detokenizer=add_detokenizer,
            **kwargs,
        )

    mlx_utils.load_processor = load_processor_with_slow_qwen_processor

    mlx_utils.convert(
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        dtype=args.dtype,
        skip_vision=args.skip_vision,
        trust_remote_code=True,
    )


if __name__ == "__main__":
    main()
