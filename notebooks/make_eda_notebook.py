import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# EDA (S5)\nRun `experiments/run_pipeline.py --download` first to create `data/processed/<basket>.npz`."
    ),
    nbf.v4.new_code_cell(
        "import sys; sys.path.insert(0, '..')\nfrom data.market_data import MarketData\nfrom data import eda\nfrom labels.r2_vol import label_r2\nBASKET = 'us_sectors'   # or 'developed_countries'\nmd = MarketData.load(f'../data/processed/{BASKET}.npz')\nlen(md), md.assets"
    ),
    nbf.v4.new_code_cell("eda.plot_returns_and_vol(md, BASKET)"),
    nbf.v4.new_code_cell("stress = label_r2(md).series\neda.plot_corr_heatmaps(md, stress, BASKET)"),
    nbf.v4.new_code_cell("eda.plot_symbol_frequencies(md, K=3, tag=BASKET)"),
]
nbf.write(nb, "eda.ipynb")
