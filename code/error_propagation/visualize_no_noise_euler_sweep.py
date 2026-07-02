from visualize_no_noise_diffrax_sweep import parse_args, visualize


def main():
    args = parse_args()
    if args.input_dir == "no_noise_diffrax_prediction_sweep_results":
        args.input_dir = "no_noise_euler_diffrax_prediction_sweep_results"
    if args.output_dir == "no_noise_diffrax_visualizations":
        args.output_dir = "no_noise_euler_diffrax_visualizations"
    visualize(args)


if __name__ == "__main__":
    main()
