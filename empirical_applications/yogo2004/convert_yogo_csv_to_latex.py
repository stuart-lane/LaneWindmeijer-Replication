import pandas as pd
import os
from config import OUTPUT_DIR

header_text = rf"""\begin{{table}}[ht!]
\small
\begin{{center}}
\begin{{threeparttable}}
\caption{{Estimates using dataset from \textcite{{yogo2004}}.}} 
\vspace{{-0.5em}}
\centering 
\begin{{tabularx}}{{\textwidth}}{{l *{{7}}{{>{{\centering\arraybackslash}}X}}}}
\hline"""

footer_text = rf"""
\hline 
\end{{tabularx}}
\vspace{{1mm}}
\caption*{{Panels (a) and (b) give estimates/test statistic values for models  $\Delta c_{{t+1}} = \mu_c + \psi r_{{t+1}} + u_{{t+1}}$ and $r_{{t+1}} = \mu_r + (1/\psi) \Delta c_{{t+1}} + \eta_{{t+1}}$ respectively. Column 3 gives the $F_{{eff}}$-statistic from \textcite{{montiel2013}}. Column 4 gives conservative simplified 95\% critical value for $F_{{eff}}$. Columns 5-6 give the estimates of $\psi$ in (a) and $1/\psi$ in (b) by 2SLS and LIML respectively. Columns 7-8 report the $J$- and $KP$-test statistic values. The critical value of interest is the 5\% significance critical value $\chi^2_{{0.95}}(3) = 7.815$, so the null hypothesis of (\ref{{eq emp hyp1}}) for Panel (a) or (\ref{{eq emp hyp2}}) for Panel (b) is rejected. Newey-West \parencite{{newey1987}} standard errors are used with $L = 6$ lags for the USA and $L=4$ lags for all other countries (the difference accounts for the longer time horizon of the USA data.)
}}
\label{{table empirical}}
\end{{threeparttable}}
\end{{center}}
\end{{table}}
"""

top_panel_text = rf"""
\hline
\multicolumn{{8}}{{c}}{{Panel (a): $\Delta c_{{t+1}} = \mu_c + \psi r_{{t+1}} + u_{{t+1}}$}} \T \\\cline{{2-3}}
\hline  
Country & $T$ &  $F_{{eff}}$ & $\kappa$ & $2SLS$ & $LIML$ & $J$ & $KP$ \T \\ 
\hline
"""

bottom_panel_text = rf"""
\hline
\multicolumn{{8}}{{c}}{{Panel (b): $r_{{t+1}} = \mu_r + (1/\psi) \Delta c_{{t+1}} + \eta_{{t+1}}$}} \T \\\cline{{2-3}}
\hline  
Country & $T$ &  $F_{{eff}}$ & $\kappa$ & $2SLS$ & $LIML$ & $J$ & $KP$ \T \\ 
\hline
"""

def row_string(
        df: pd.DataFrame, 
        row: int
    ) -> str:
    """Generate row of data as string"""
    
    row_info = df.iloc[row]

    row_string = f"{row_info["Country"]} & {row_info["n"]} & {row_info["F"]:.2f} & {row_info["cv"]:.2f} & {row_info["2SLS"]:.2f} & {row_info["LIML"]:.2f} & {row_info["J"]:.2f} & {row_info["KP"]:.2f} \\\\"
    return row_string

def generate_panel(
        df: pd.DataFrame
    ) -> str:
    """Create panel of table"""

    number_of_rows = len(df)
    panel_list = []

    for row_idx in range(0, number_of_rows):
        panel_list.append(row_string(df, row_idx))

    return "\n".join(panel_list)

def table_raw_text(
        df_1: pd.DataFrame, 
        df_2: pd.DataFrame
    ) -> str:
    """Generate complete latex table"""

    full_table = []
    full_table.append(header_text)
    full_table.append(top_panel_text)
    full_table.append(generate_panel(df_1))
    full_table.append(bottom_panel_text)
    full_table.append(generate_panel(df_2))
    full_table.append(footer_text)
    table = "".join(full_table)

    return table

def main():
    df_1 = pd.read_csv(os.path.join(OUTPUT_DIR, "table_1.csv"))
    df_2 = pd.read_csv(os.path.join(OUTPUT_DIR, "table_2.csv"))

    table = table_raw_text(df_1, df_2)

    # Path to save the file
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    file_path = f"./{OUTPUT_DIR}/table_6_1.tex"

    # Write row to file
    with open(file_path, "w") as f:
        f.write(table + "\n")

    print(f"Table saved to {file_path}")

if __name__ == "__main__":
    main()
