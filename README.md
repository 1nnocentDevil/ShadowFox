👤 About Me
Hi! I'm a Data Science Intern at ShadowFox, currently working through a structured learning path focused on Python-based data visualization. This repository documents my internship tasks, code submissions, and learning progress across the beginner, intermediate, and advanced levels.

🏢 About ShadowFox
ShadowFox is a forward-thinking organization that empowers aspiring data professionals through hands-on, project-based internship programs. The Data Science stream covers the full pipeline — from data wrangling and visualization to machine learning and model deployment.

📁 Repository Structure
shadowfox-ds-internship/
│
├── 📂 Beginner_Level/
│   ├── matplotlib_charts/
│   │   ├── line_plot.py
│   │   ├── scatter_plot.py
│   │   ├── bar_chart.py
│   │   ├── histogram.py
│   │   ├── pie_chart.py
│   │   ├── box_plot.py
│   │   └── heatmap.py
│   ├── seaborn_charts/
│   │   ├── scatter_relational.py
│   │   ├── line_plot_ci.py
│   │   ├── bar_plot.py
│   │   ├── histogram_kde.py
│   │   ├── box_plot.py
│   │   ├── violin_plot.py
│   │   └── heatmap.py
│   └── README.md  ← You are here
│
├── 📂 Intermediate_Level/     (coming soon)
├── 📂 Advanced_Level/         (coming soon)
│
└── README.md

✅ Task 1 — Beginner Level: Python Visualization Libraries
📌 Objective
Demonstrate proficiency in Matplotlib and Seaborn by implementing 7 chart types per library with live rendered examples and ready-to-run code snippets.

📊 Matplotlib — Charts Implemented
#Chart TypeDescriptionUse Case1Line PlotConnects data points with a continuous lineStock prices, temperature trends, loss curves2Scatter PlotIndividual points revealing correlations/clustersCorrelation analysis, outlier detection3Bar ChartRectangular bars comparing discrete categoriesQuarterly revenue, survey results4HistogramBins continuous data to show frequencyDistribution analysis, normality checks5Pie ChartProportional slices of a wholeMarket share, budget allocation6Box PlotQuartile-based distribution summaryScore comparisons, outlier spotting7HeatmapMatrix values encoded as colorsCorrelation matrices, activity maps

📊 Seaborn — Charts Implemented
#Chart TypeDescriptionUse Case1Scatter / Relational PlotTwo continuous variables with hue/size encodingsMulti-dimensional EDA2Line Plot with CIAuto-computed confidence intervals on trendsTime-series with uncertainty3Bar PlotGroup means with bootstrapped error barsA/B testing, category benchmarks4Histogram + KDEDistribution view with density curve overlayNormality checks, group comparison5Box PlotRichly styled quartile plots with auto-groupingDistribution comparisons6Violin PlotBox plot + KDE combinedBimodal data, publication figures7HeatmapDataFrame rendered as annotated colored gridCorrelation matrices, pivot tables

🛠️ Libraries & Tools Used
python# Core Libraries
import matplotlib.pyplot as plt   # Foundational plotting (created 2003, John D. Hunter)
import seaborn as sns              # Statistical visualization (built on Matplotlib)
import numpy as np                 # Numerical arrays
import pandas as pd                # DataFrames (Seaborn's native format)
from scipy.stats import norm       # Statistical distributions
Installation:
bashpip install matplotlib seaborn numpy pandas scipy

⚡ Quick Sample — Matplotlib Line Plot
pythonimport matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 200)
plt.figure(figsize=(8, 4))
plt.plot(x, np.sin(x), color='steelblue', lw=2, label='sin(x)')
plt.plot(x, np.cos(x), color='#e65100', lw=2, ls='--', label='cos(x)')
plt.title('Line Plot — Sine & Cosine')
plt.xlabel('X'); plt.ylabel('Amplitude')
plt.legend(); plt.grid(True, alpha=0.4)
plt.tight_layout(); plt.show()

⚡ Quick Sample — Seaborn Violin Plot
pythonimport seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style='whitegrid')
tips = sns.load_dataset('tips')
plt.figure(figsize=(7, 4))
sns.violinplot(data=tips, x='day', y='tip',
               hue='sex', split=True, inner='quart',
               palette={'Male': '#4472c4', 'Female': '#ed7d31'},
               order=['Thur', 'Fri', 'Sat', 'Sun'])
plt.title('Tip Distribution by Day & Gender')
plt.tight_layout(); plt.show()

🔍 Matplotlib vs Seaborn — Key Differences
DimensionMatplotlibSeabornEase of UseSteeper learning curveBeginner-friendlyCode VerbosityMore explicit, more linesConcise, high-level APIDefault AestheticsPlain; needs manual stylingBeautiful out of the boxStatistical ToolsNone built-inAuto CI, KDE, regressionPandas SupportManual Series/arraysNative data= parameterBest ForCustom layouts, full controlEDA, statistical, quick visuals

Rule of thumb: Use Seaborn for fast, attractive statistical plots. Use Matplotlib when you need fine-grained control, animations, or multi-panel layouts.


📈 Learning Progress

 Beginner Level — Matplotlib & Seaborn (14 chart types)
 Intermediate Level — Pandas, Plotly, EDA pipelines
 Advanced Level — ML visualizations, dashboards, storytelling


📚 References

Matplotlib Quick Start
Matplotlib Gallery
Seaborn Introduction
Seaborn API Reference
