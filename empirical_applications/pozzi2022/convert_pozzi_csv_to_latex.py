import pandas as pd
import os
from config import OUTPUT_DIR

header_text = rf"""\begin{{table}}[ht!]
\small
\begin{{center}}
\begin{{threeparttable}}
\caption{{Estimates using dataset from \textcite{{pozzi2022}}}} 
\vspace{{-0.5em}}
\centering 
\begin{{tabularx}}{{\textwidth}}{{l *{{7}}{{>{{\centering\arraybackslash}}X}}}}
\hline"""

footer_text = rf""""
\hline 
\end{{tabularx}}
\vspace{{1mm}}
\caption*{{Panels (a) and (b) give estimates/test statistic values for models using the instrument sets $Z_{{t+1}}' = (\,  r_{{t}},\, \bar{{r}}_{{t+1}}^{{for}} \,)$ and $Z_{{t+1}}' = (\,  r_{{t-1}},\, \bar{{r}}_{{t+1}}^{{for}} \,)$ respectively. Column 3 gives the $F_{{eff}}$-statistic from \textcite{{montiel2013}}. Column 4 gives conservative simplified 95\% critical value for $F_{{eff}}$.  Columns 5-6 give the estimates of $\psi$ by 2SLS and LIML respectively. Columns 7-8 report the $J$- and $KP$-test statistic values. The critical value of interest is the 5\% significance critical value $\chi^2_{{0.95}}(1) = 3.841$. Newey-West standard errors are calculated with $L = 4$ lags \parencite{{newey1987}}.
}}
\label{{table empiricalpozzi}}
\end{{threeparttable}}
\end{{center}}
\end{{table}}
"""

top_panel_text = rf"""
\hline
\multicolumn{{7}}{{c}}{{Panel (a): Instrument set $Z_{{t+1}}' = (\,  r_{{t}},\, \bar{{r}}_{{t+1}}^{{for}} \,)$}} \T \\\cline{{2-3}}
\hline
Country & $T$ & $F_{{eff}}$ & $\kappa$ & $2SLS$ & $LIML$ & $J$ & $KP$ \\ 
\hline 
"""

bottom_panel_text = rf"""
\hline
\multicolumn{{7}}{{c}}{{Panel (b): Instrument set $Z_{{t+1}}' = (\,  r_{{t-1}},\, \bar{{r}}_{{t+1}}^{{for}} \,)$}} \T \\\cline{{2-3}}
\hline
Country & $T$ & $F_{{eff}}$ & $\kappa$ & $2SLS$ & $LIML$ & $J$ & $KP$ \\ 
\hline 
"""

COUNTRY_MAP = {
    "DNK": "DEN",
    "NLD": "NTH",
    "PRT": "POR",
    "ESP": "SPA",
    "CHE": "SWT",
    "GBR": "UK",
}

SKIP_COUNTRIES = {"DEU", "JPN"}

def row_string(
        df: pd.DataFrame, 
        row: int
    ) -> str:
    """Generate row of data as string"""

    row_info = df.iloc[row]

    country = row_info["Country"]

    # skip unwanted countries
    if country in SKIP_COUNTRIES:
        return None

    # apply mapping if it exists
    country = COUNTRY_MAP.get(country, country)

    return f"{country} & {row_info['n']} & {row_info['F']:.2f} & {row_info['cv']:.2f} & {row_info['2SLS']:.2f} & {row_info['LIML']:.2f} & {row_info['J']:.2f} & {row_info['KP']:.2f} \\\\ "

def generate_panel(
        df: pd.DataFrame
    ) -> str:
    """Create panel of table"""

    panel_list = []

    for row_idx in range(len(df)):
        row = row_string(df, row_idx)
        if row is not None:
            panel_list.append(row)

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

    file_path = f"./{OUTPUT_DIR}/table_A_1.tex"

    # Write row to file
    with open(file_path, "w") as f:
        f.write(table + "\n")

    print(f"Table saved to {file_path}")

if __name__ == "__main__":
    main()