import os
import pandas as pd

# OUTPUT_FOLDER = "python/simulations/output"
OUTPUT_FOLDER = "output"
KZ_VALUES = [2,4]
MU2_VALUES = [1, 4, 8, 16]

header_text = rf"""\begin{{table}}[ht!]
\small
\begin{{center}}
\begin{{threeparttable}}
\caption[Estimation and test statistic results for $k_z - k_x = 1$]{{Estimation and test statistic results for $k_z - k_x = 1$.}} 
\vspace{{-0.5em}}
\centering 
\begin{{tabularx}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}} c c c c c c c c c c}}
\hline"""

column_names = rf"""& & \multicolumn{{2}}{{c}}{{Median Bias}} & \multicolumn{{2}}{{c}}{{90:10 Range}} & \multicolumn{{2}}{{c}}{{Nom. Size 10\%}} &  \multicolumn{{2}}{{c}}{{Nom. Size 1\%}}  \T \\\cline{{2-3}}
\hline
$\alpha$ & $\mu^2$ & 2SLS & LIML & 2SLS & LIML & $J$ & $KP$ & $J$ & $KP$ \\ 
\hline
\hline"""

footer_text = rf"""\end{{tabularx}}
\caption*{{\footnotesize{{Additional estimator and test statistic results for 2SLS, LIML, $J$ and $KP$ with one overidentifying restriction. Median bias and 90:10 range (defined as the $90^{{th}}$ percentile minus the $10^{{th}}$ percentile) are reported for 2SLS and LIML. Rejection frequencies at the nominal 10\% and 1\% levels are reported for $J$ and $KP$.}}}}
\label{{table extraresults1}}
\end{{threeparttable}}
\end{{center}}
\end{{table}}
"""

def generate_panel(
        df: pd.DataFrame, 
        kz: int, 
        rho: float, 
        alpha: float
    ) -> str:
    """Create panel of table"""

    row_idx = 0
    panel = []
    for mu2 in MU2_VALUES:
        info = df.loc[(df['kz'] == kz) & (df['alpha'] == alpha) & (df['rho'] == rho) & (df['mu2'] == mu2)]
        row = info.iloc[0]
        
        row_string = (
            f"& {mu2} "
            f"& {row['TSLS_median_bias']:.3f} "
            f"& {row['LIML_median_bias']:.3f} "
            f"& {row['TSLS_ninety_ten']:.3f} "
            f"& {row['LIML_ninety_ten']:.3f} "
            f"& {row['J_reject_10']:.3f} "
            f"& {row['KP_reject_10']:.3f} "
            f"& {row['J_reject_1']:.3f} "
            f"& {row['KP_reject_1']:.3f} \\\\"
        )

        if row_idx % 4 == 0:
            row_string = rf"\multirow{{4}}{{*}}{{{alpha}}} " + row_string

        panel.append(row_string)
        row_idx += 1

    panel.append(r"\hline")
    return "\n".join(panel)

def create_subpanel_header(
        rho: float
    ) -> str:
    """Generate headers for endogeneity types"""

    if rho == 0.2:
        endogeneity_tag = "Low"
    elif rho == 0.5: 
        endogeneity_tag = "Medium"
    else:
        endogeneity_tag = "High"

    subpanel_header_text = rf"""\multicolumn{{10}}{{c}}{{{endogeneity_tag} endogeneity design ($\rho$ = {rho})}} \\
\hline"""

    return subpanel_header_text

def create_table(
        df: pd.DataFrame, 
        kz: int
    ) -> str:
    """Generate complete latex table"""

    table = []

    table.append(header_text)
    table.append(create_subpanel_header(rho=0.2))
    table.append(column_names)
    table.append(generate_panel(df, kz=kz, rho=0.2, alpha=0.5))
    table.append(generate_panel(df, kz=kz, rho=0.2, alpha=1))
    table.append(generate_panel(df, kz=kz, rho=0.2, alpha=2))
    table.append(create_subpanel_header(rho=0.5))
    table.append(column_names)
    table.append(generate_panel(df, kz=kz, rho=0.5, alpha=0.5))
    table.append(generate_panel(df, kz=kz, rho=0.5, alpha=1))
    table.append(generate_panel(df, kz=kz, rho=0.5, alpha=2))
    table.append(create_subpanel_header(rho=0.95))
    table.append(column_names)
    table.append(generate_panel(df, kz=kz, rho=0.95, alpha=0.5))
    table.append(generate_panel(df, kz=kz, rho=0.95, alpha=1))
    table.append(generate_panel(df, kz=kz, rho=0.95, alpha=2))
    table.append(footer_text)

    return "\n".join(table)

def main():

    df = pd.read_csv(os.path.join(OUTPUT_FOLDER, "simulation_metrics_mu2_varying.csv"))

    for kz in KZ_VALUES:
        table = create_table(df, kz=kz)

        # Path to save the file
        if not os.path.exists(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER)

        file_path = f"./{OUTPUT_FOLDER}/table_D_{int(kz/2)}.tex"

        # Write row to file
        with open(file_path, "w") as f:
            f.write(table + "\n")

        print(f"Table saved to {file_path}")

if __name__ == "__main__":
    main()
