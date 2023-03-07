from Handler.data_loader import load_data_kidz
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
import sys
import pandas as pd


def load_model(path_model):
    model = torch.load(path_model)
    return model


def main(path_training_data, path_model, path_save_plots, number_samples, plot_chain, plot_luptize_conditions,
         conditions, plot_luptize_input, plot_color_color, bands, colors):
    col_label_flow = [
        "axis_ratio_input",
        "Re_input",
        "sersic_n_input",
        "u_input",
        "g_input",
        "r_input",
        "i_input",
        "Z_input",
        "Y_input",
        "J_input",
        "H_input",
        "Ks_input",
        "InputSeeing_u",
        "InputSeeing_g",
        "InputSeeing_r",
        "InputSeeing_i",
        "InputSeeing_Z",
        "InputSeeing_Y",
        "InputSeeing_J",
        "InputSeeing_H",
        "InputSeeing_Ks",
        "InputBeta_u",
        "InputBeta_g",
        "InputBeta_r",
        "InputBeta_i",
        "InputBeta_Z",
        "InputBeta_Y",
        "InputBeta_J",
        "InputBeta_H",
        "InputBeta_Ks",
        "rmsAW_u",
        "rmsAW_g",
        "rmsAW_r",
        "rmsAW_i",
        "rms_Z",
        "rms_Y",
        "rms_J",
        "rms_H",
        "rms_Ks"
    ]

    col_output_flow = [
        "luptize_u",
        "luptize_g",
        "luptize_r",
        "luptize_i",
        "luptize_Z",
        "luptize_Y",
        "luptize_J",
        "luptize_H",
        "luptize_Ks",
        "luptize_err_u",
        "luptize_err_g",
        "luptize_err_r",
        "luptize_err_i",
        "luptize_err_Z",
        "luptize_err_Y",
        "luptize_err_J",
        "luptize_err_H",
        "luptize_err_Ks",
        "FLUX_AUTO",
        "FLUXERR_AUTO",
    ]
    print("Load data...")
    train_data, valid_data, test_data = load_data_kidz(
        path_training_data=path_training_data,
        input_flow=col_label_flow,
        output_flow=col_output_flow,
        selected_scaler="MaxAbsScaler",
        apply_cuts=True
    )
    scaler = test_data["scaler"]

    # Write data as torch loader
    test_tensor = torch.from_numpy(test_data[f"output flow in order {col_output_flow}"])
    test_labels = torch.from_numpy(test_data[f"label flow in order {col_label_flow}"])
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=number_samples,
        shuffle=False,
        drop_last=False,
        # **kwargs
    )

    print("Load model...")
    model = load_model(path_model)
    for batch_idx, data in enumerate(test_loader):
        cond_data = data[1].float()
        with torch.no_grad():
            print("Sample data")
            test_output = model.sample(num_samples=number_samples, cond_inputs=cond_data).detach()
        print("Plot data")
        plot_data(
            path_save_plots=path_save_plots,
            cond_data=cond_data,
            test_output=test_output,
            col_label_flow=col_label_flow,
            col_output_flow=col_output_flow,
            scaler=scaler,
            data=data[0],
            plot_chain=plot_chain,
            plot_luptize_conditions=plot_luptize_conditions,
            conditions=conditions,
            plot_luptize_input=plot_luptize_input,
            plot_color_color=plot_color_color,
            bands=bands,
            colors=colors,
        )
        break


def plot_data(path_save_plots, cond_data, test_output, col_label_flow, col_output_flow, scaler, data, plot_chain,
              plot_luptize_conditions, conditions, plot_luptize_input, plot_color_color, bands, colors):

    df_generator_label = pd.DataFrame(cond_data.numpy(), columns=col_label_flow)
    df_generator_output = pd.DataFrame(test_output.numpy(), columns=col_output_flow)
    df_generator_scaled = pd.concat([df_generator_label, df_generator_output], axis=1)
    generator_rescaled = scaler.inverse_transform(df_generator_scaled)
    df_generated = pd.DataFrame(generator_rescaled, columns=df_generator_scaled.columns)

    df_true_output = pd.DataFrame(data, columns=col_output_flow)
    df_true_scaled = pd.concat([df_generator_label, df_true_output], axis=1)
    true_rescaled = scaler.inverse_transform(df_true_scaled)
    df_true = pd.DataFrame(true_rescaled, columns=df_true_scaled.columns)

    if plot_chain is True:
        df_generated_measured = pd.DataFrame({
            "luptize_u": np.array(df_generated["luptize_u"]),
            "luptize_r": np.array(df_generated["luptize_r"]),
            "luptize_g": np.array(df_generated["luptize_g"]),
            "luptize_i": np.array(df_generated["luptize_i"]),
            "luptize_Z": np.array(df_generated["luptize_Z"]),
            "luptize_Y": np.array(df_generated["luptize_Y"]),
            "luptize_J": np.array(df_generated["luptize_J"]),
            "luptize_H": np.array(df_generated["luptize_H"]),
            "luptize_Ks": np.array(df_generated["luptize_Ks"])
        })

        df_true_measured = pd.DataFrame({
            "luptize_u": np.array(df_true["luptize_u"]),
            "luptize_r": np.array(df_true["luptize_r"]),
            "luptize_g": np.array(df_true["luptize_g"]),
            "luptize_i": np.array(df_true["luptize_i"]),
            "luptize_Z": np.array(df_true["luptize_Z"]),
            "luptize_Y": np.array(df_true["luptize_Y"]),
            "luptize_J": np.array(df_true["luptize_J"]),
            "luptize_H": np.array(df_true["luptize_H"]),
            "luptize_Ks": np.array(df_true["luptize_Ks"])
        })

        arr_true = df_true_measured.to_numpy()
        arr_generated = df_generated_measured.to_numpy()
        parameter = [
            "luptize u",
            "luptize r",
            "luptize g",
            "luptize i",
            "luptize Z",
            "luptize Y",
            "luptize J",
            "luptize H",
            "luptize Ks",
        ]
        chainchat = ChainConsumer()
        chainchat.add_chain(arr_true, parameters=parameter, name="true observed properties: chat")
        chainchat.add_chain(arr_generated, parameters=parameter, name="generated observed properties: chat*")
        chainchat.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
        chainchat.plotter.plot(
            filename=f'{path_save_plots}/chainplot.png',
            figsize="page",
        )
        # plt.show()
        plt.clf()

    if plot_luptize_conditions is True:
        for condition in conditions:
            df_lineplot_u = pd.DataFrame({
                condition: df_true[condition],
                "measured luptize - generated luptize": df_true["luptize_u"]-df_generated["luptize_u"],
                "band": ["u" for _ in range(len(df_true[condition]))]
            })
            df_lineplot_g = pd.DataFrame({
                condition: df_true[condition],
                "measured luptize - generated luptize": df_true["luptize_g"] - df_generated["luptize_g"],
                "band": ["g" for _ in range(len(df_true[condition]))]
            })
            df_lineplot_r = pd.DataFrame({
                condition: df_true[condition],
                "measured luptize - generated luptize": df_true["luptize_r"] - df_generated["luptize_r"],
                "band": ["r" for _ in range(len(df_true[condition]))]
            })

            df_lineplot_i = pd.DataFrame({
                condition: df_true[condition],
                "measured luptize - generated luptize": df_true["luptize_i"] - df_generated["luptize_i"],
                "band": ["i" for _ in range(len(df_true[condition]))]
            })
            df_lineplot_Z = pd.DataFrame({
                condition: df_true[condition],
                "measured luptize - generated luptize": df_true["luptize_Z"] - df_generated["luptize_Z"],
                "band": ["Z" for _ in range(len(df_true[condition]))]
            })
            df_lineplot_Y = pd.DataFrame({
                condition: df_true[condition],
                "measured luptize - generated luptize": df_true["luptize_Y"] - df_generated["luptize_Y"],
                "band": ["Y" for _ in range(len(df_true[condition]))]
            })
            df_lineplot_J = pd.DataFrame({
                condition: df_true[condition],
                "measured luptize - generated luptize": df_true["luptize_J"] - df_generated["luptize_J"],
                "band": ["J" for _ in range(len(df_true[condition]))]
            })
            df_lineplot_H = pd.DataFrame({
                condition: df_true[condition],
                "measured luptize - generated luptize": df_true["luptize_H"] - df_generated["luptize_H"],
                "band": ["H" for _ in range(len(df_true[condition]))]
            })
            df_lineplot_Ks = pd.DataFrame({
                condition: df_true[condition],
                "measured luptize - generated luptize": df_true["luptize_Ks"] - df_generated["luptize_Ks"],
                "band": ["Ks" for _ in range(len(df_true[condition]))]
            })


            df_lineplot = pd.concat(
                [
                    df_lineplot_u,
                    df_lineplot_g,
                    df_lineplot_r,
                    df_lineplot_i,
                    df_lineplot_Z,
                    df_lineplot_Y,
                    df_lineplot_J,
                    df_lineplot_H,
                    df_lineplot_Ks,
                ],
                ignore_index=True,
                sort=False
            )

            sns.lineplot(data=df_lineplot, x=condition, y="measured luptize - generated luptize", hue="band")
            plt.savefig(f"{path_save_plots}/luptize_{condition}_plot.png")
            # plt.show()
            plt.clf()

        if plot_luptize_input is True:
            for band in bands:
                df_lineplot_input_true = pd.DataFrame({
                    f"{band}_input": df_true[f"{band}_input"],
                    "true - luptize": df_true[f"{band}_input"] - df_true[f"luptize_{band}"],
                    "dataset": ["skills" for _ in range(len(df_true[f"{band}_input"]))]
                })

                df_lineplot_input_generated = pd.DataFrame({
                    f"{band}_input": df_true[f"{band}_input"],
                    "true - luptize": df_true[f"{band}_input"] - df_generated[f"luptize_{band}"],
                    "dataset": ["generated" for _ in range(len(df_true[f"{band}_input"]))]
                })

                df_lineplot_input = pd.concat(
                    [
                        df_lineplot_input_true,
                        df_lineplot_input_generated
                    ],
                    ignore_index=True,
                    sort=False
                )

                sns.displot(data=df_lineplot_input, x=f"{band}_input", y="true - luptize", hue="dataset")
                plt.savefig(f"{path_save_plots}/luptize_input_{band}.plot.png")
                # plt.show()
                plt.clf()

        if plot_color_color is True:
            for color in colors:
                df_color_color_generated = pd.DataFrame({
                    f"luptize {color[0]} - luptize {color[1]}": df_generated[f"luptize_{color[0]}"] - df_generated[f"luptize_{color[1]}"],
                    f"luptize {color[1]} - luptize {color[2]}": df_generated[f"luptize_{color[1]}"] - df_generated[f"luptize_{color[2]}"],
                    "dataset": ["generated" for _ in range(len(df_generated[f"luptize_{color[0]}"]))]
                })
                df_color_color_skills = pd.DataFrame({
                    f"luptize {color[0]} - luptize {color[1]}": df_true[f"luptize_{color[0]}"] - df_true[f"luptize_{color[1]}"],
                    f"luptize {color[1]} - luptize {color[2]}": df_true[f"luptize_{color[1]}"] - df_true[f"luptize_{color[2]}"],
                    "dataset": ["skills" for _ in range(len(df_true[f"luptize_{color[0]}"]))]
                })

                df_color_color = pd.concat(
                    [
                        df_color_color_generated,
                        df_color_color_skills
                    ],
                    ignore_index=True,
                    sort=False
                )

                sns.displot(data=df_color_color, x=f"luptize {color[0]} - luptize {color[1]}", y=f"luptize {color[1]} - luptize {color[2]}", hue="dataset")
                plt.savefig(f"{path_save_plots}/color_color_{color}.plot.png")
                # plt.show()
                plt.clf()



if __name__ == '__main__':
    path = os.path.abspath(sys.path[0])
    lst_conditions = [
        "sersic_n_input"
    ]
    lst_bands = [
        "u",
        "g",
        "r",
        "i",
        "Z",
        "Y",
        "J",
        "H",
        "Ks"
    ]
    lst_colors = [
        ("u", "g", "r"),
        ("g", "r", "i"),
        ("r", "i", "Z"),
        ("i", "Z", "Y"),
        ("Z", "Y", "J"),
        ("Y", "J", "H"),
        ("J", "H", "Ks"),
    ]
    main(
        path_training_data=f"{path}/../Data/kids_training_catalog_lup.pkl",
        path_model=f"{path}/../trained_models/best_model_epoch_100.pt",
        path_save_plots=f"{path}/output_run_flow",
        number_samples=1500000,
        plot_chain=False,
        plot_luptize_conditions=True,
        conditions=lst_conditions,
        plot_luptize_input=True,
        plot_color_color=True,
        bands=lst_bands,
        colors=lst_colors
    )

