"""Plotting functions."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from statannot import add_stat_annotation
import warnings
warnings.filterwarnings('ignore')


def pval_df(pmethod):
    """
    Return lambda function to create p-value dataframes.

    Parameters
    ----------
    pmethod : string
        Which method to use for the p-values.
        Options: "pearson", "spearman", "kendall".

    Returns
    -------
    lambda function
        Function to calculate p-value.
    """
    if pmethod == "pearson":
        return lambda x, y: pearsonr(x, y)[1]
    if pmethod == "spearman":
        return lambda x, y: spearmanr(x, y)[1]
    if pmethod == "kendall":
        return lambda x, y: kendalltau(x, y)[1]

    
def fig13(df):
    plt.rcParams.update(plt.rcParamsDefault)
    fontsize = 14
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["axes.labelsize"] = fontsize
    plt.rcParams["axes.labelsize"] = fontsize
    plt.rcParams["lines.linewidth"] = 3
    plt.rcParams["lines.markersize"] = fontsize*.7
    plt.rcParams['xtick.major.size'] = fontsize*.3
    plt.rcParams['xtick.major.width'] = fontsize*.1
    plt.rcParams['ytick.major.size'] = fontsize*.3
    plt.rcParams['ytick.major.width'] = fontsize*.1
    
    colors = ["purple", "darkorange", "k", "g"]
    markers = ["+", "s", "^", "o"]
    
    df_single_EZ = df.loc[df.num_EZ == 1]
    # restructure df to account for duplicate seeg and clin areas
    df_noEZ = df[~df.EZ]
    df_noEZ["kind_seeg"] = "other"
    df_noEZ["kind_clin"] = "other"
    df_noEZ.loc[(df_noEZ.PZ_seeg), ("kind_seeg")] = r"$PZ_{SEEG}$"
    df_noEZ.loc[(df_noEZ.PZ_clin), ("kind_clin")] = r"$PZ_{Clin}$"
    
    _, ax = plt.subplots(3, 2, figsize=(12, 10), sharey="row")
    for pat, color, marker in zip(df_single_EZ.patient.unique(), colors, markers):
        df[df.patient==pat].plot(x="short_EZ",
                                 y="rec_time", 
                                 kind="scatter",
                                 c=color,
                                 s=30,
                                 label=pat,
                                 marker=marker,
                                 legend=False,
                                 ax=ax[0, 0])
        df[df.patient==pat].plot(x="weight_EZ_log",
                                 y="rec_time", 
                                 kind="scatter",
                                 c=color,
                                 s=30,
                                 label=pat,
                                 marker=marker,
                                 ax=ax[0, 1])
    
    sns.scatterplot(x="short_ord",
                    y="order",
                    data=df_noEZ,
                    hue="kind",
                    hue_order=["PZ_seeg", "PZ_clin", "other"],
                    palette=["deepskyblue", "darkorange", "grey"],
                    size=df_noEZ.plot_size,
                    sizes=(8, 40),
                    legend=False,
                    ax=ax[1, 0])
    sns.scatterplot(x="short_ord",
                    y="order",
                    data=df_noEZ[(df_noEZ.kind == "PZ_seeg") | (df_noEZ.kind == "PZ_clin")],
                    hue="kind",
                    hue_order=["PZ_seeg", "PZ_clin"],
                    palette=["deepskyblue", "darkorange"],
                    size=df_noEZ.plot_size,
                    sizes=(8, 40),
                    edgecolor="k",
                    linewidth=.2,
                    facecolor=None,
                    legend=False,
                    ax=ax[1, 0])
    
    sns.scatterplot(x="weight_ord",
                    y="order",
                    data=df_noEZ,
                    hue="kind",
                    hue_order=["PZ_seeg", "PZ_clin", "other"],
                    palette=["deepskyblue", "darkorange", "grey"],
                    size=df_noEZ.plot_size,
                    sizes=(8, 40),
                    legend="full",
                    ax=ax[1, 1])
    sns.scatterplot(x="weight_ord",
                    y="order",
                    data=df_noEZ[(df_noEZ.kind == "PZ_seeg") | (df_noEZ.kind == "PZ_clin")],
                    hue="kind",
                    hue_order=["PZ_seeg", "PZ_clin"],
                    palette=["deepskyblue", "darkorange"],
                    size=df_noEZ.plot_size,
                    sizes=(8, 40),
                    edgecolor="k",
                    linewidth=.2,
                    facecolor=None,
                    legend="full",
                    ax=ax[1, 1])
    
    loc_legend = (1.05, 0.4)
    ax[0, 0].set_xlabel("Shortest Path (Value)")
    ax[0, 0].set_ylabel("$t_{rec}$(s)")
    ax[0, 1].set_xlabel("Logarithmic Weight (Value)")
    ax[0, 1].legend(loc=loc_legend)
    ax[1, 0].set_xlabel("Shortest Path (Order)")
    ax[1, 0].set_ylabel("$t_{rec}$ (Order)")
    ax[1, 1].set_xlabel("Weight to EZ (Order)")
    ax[1, 1].set_ylabel("$t_{rec}$ (Order)")
    ax[1, 1].invert_xaxis()
    handles, labels = ax[1, 1].get_legend_handles_labels()
    labels = ["$PZ_{SEEG}$", "$PZ_{Clin}$", "other"]
    ax[1, 1].legend(handles=handles[1:-3], labels=labels, loc=loc_legend)
    
    sns.scatterplot(x="short_EZ",
                    y="rec_time",
                    data=df_noEZ.loc[df_noEZ.PZ_seeg],
                    hue="patient",
                    style="patient",
                    s=100,
                    legend=False,
                    ax=ax[2, 0])
    sns.scatterplot(x="short_EZ",
                    y="rec_time",
                    data=df_noEZ.loc[df_noEZ.PZ_seeg],
                    hue="patient",
                    style="patient",
                    s=100,
                    facecolor=None,
                    edgecolor="k",
                    linewidth=.2,
                    legend=False,
                    ax=ax[2, 0])
    
    sns.scatterplot(x="short_EZ",
                    y="rec_time",
                    data=df_noEZ.loc[df_noEZ.PZ_clin],
                    hue="patient",
                    style="patient",
                    s=100,
                    legend="full",
                    ax=ax[2, 1])
    sns.scatterplot(x="short_EZ",
                    y="rec_time",
                    data=df_noEZ.loc[df_noEZ.PZ_clin],
                    hue="patient",
                    style="patient",
                    s=100,
                    facecolor=None,
                    edgecolor="k",
                    linewidth=.2,
                    legend=False,
                    ax=ax[2, 1])
    
    ax[2, 0].set_xlabel("Shortest path $PZ_{SEEG}$")
    ax[2, 0].set_ylabel("$t_{rec}(s)$")
    ax[2, 1].set_xlabel("Shortest path $PZ_{Clin}$")
    ax[2, 1].set_ylabel("$t_{rec}$")
    handles, labels = ax[2, 1].get_legend_handles_labels()
    ax[2, 1].legend(handles, labels, loc=(1.05, 0), ncol=2, columnspacing=.5,
                    labelspacing=.74, handletextpad=0.00001, fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=.35)
    panel_dic = dict(x=0, y=1.02, fontweight="bold")
    plt.text(s="A", **panel_dic, transform=ax[0, 0].transAxes)
    plt.text(s="B", **panel_dic, transform=ax[0, 1].transAxes)
    plt.text(s="C", **panel_dic, transform=ax[1, 0].transAxes)
    plt.text(s="D", **panel_dic, transform=ax[1, 1].transAxes)
    plt.text(s="E", **panel_dic, transform=ax[2, 0].transAxes)
    plt.text(s="F", **panel_dic, transform=ax[2, 1].transAxes)
    plt.show()


def fig14(df):
    plt.rcParams.update(plt.rcParamsDefault)
    # Sort pats by median shortest path length to the EZ
    median_short_EZ = [df.loc[df.patient == pat, "short_EZ"].median()
                       for pat in df.patient.unique()]
    pats_median = [patients for _, patients 
                 in sorted(zip(median_short_EZ, df.patient.unique()))]
    
    g = sns.catplot("rec_time",
                    "patient",
                    data=df.loc[df.rec_time > 0],
                    kind="box",
                    height=10,
                    order=pats_median)
    
    sns.swarmplot("rec_time",
                  "patient",
                  data=df.loc[df.rec_time > 0],
                  alpha=0.6,
                  color="black",
                  ax=g.ax,
                  size=3,
                  order=pats_median)
    plt.xlabel("$t_{rec}$(s)", fontsize=20)
    plt.ylabel(None)
    plt.xticks(np.arange(0, 1, .1), fontsize=15)
    plt.yticks(plt.gca().get_yticks(), fontsize=15)
    plt.show()


def fig20(df):
    plt.rcParams.update(plt.rcParamsDefault)
    # Plotting dics
    boxprops = dict(color="w", edgecolor=(0, 0, 0, 1), zorder=3,
                    facecolor=(0, 0, 0, 0), lw=1.5)
    medianprops = dict(color="black", lw=1.5)
    stripdic = dict(y="rec_time", alpha=1, jitter=0.35, edgecolor="k",
                    linewidth=.4)
    boxdic = dict(y="rec_time", boxprops=boxprops, fliersize=0,
                  zorder=3, medianprops=medianprops, whiskerprops=medianprops)
    statsdic = dict(y="rec_time", test='Mann-Whitney-ls',
                    comparisons_correction=None, text_format='simple',
                    loc='inside', verbose=2)
    griddic = dict(nrows=4, ncols=2, wspace=0, hspace=0.35)
    
    # Make legend on dummy plots
    leg_dic = dict(edgecolor="k", linewidth=.4)
    colors = ["deepskyblue", "lightgrey", "darkorange", "w"]
    labels = ['$PZ_{SEEG}$', 'areas without $PZ_{SEEG}$', '$PZ_{Clin}$',
              'areas without $PZ_{Clin}$']
    for color, label in zip(colors, labels):
        plt.scatter(0, 0, c=color, label=label, **leg_dic)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.close()
    
    # sort patients for correct order in loop
    pat_order = [el for num in range(5) for el in range(num, 13+num, 4)]      
    
    # restructure df to account for duplicate seeg and clin areas
    df_noEZ = df[~df.EZ]
    df_noEZ["kind_seeg"] = "other"
    df_noEZ["kind_clin"] = "other"
    df_noEZ.loc[(df_noEZ.PZ_seeg), ("kind_seeg")] = r"$PZ_{SEEG}$"
    df_noEZ.loc[(df_noEZ.PZ_clin), ("kind_clin")] = r"$PZ_{Clin}$"
    patients = df.patient.unique()
     
    fig = plt.figure(figsize=[11, 11])
    
    space_left = 0.03
    space_right = 0.18
    pat = 0
    for i in range(4):
        gs = fig.add_gridspec(left=space_left, right=space_right, **griddic)
        for j in range(4):
            if i == 3 and j == 3:
                ax = fig.add_subplot(gs[j, 0])
                ax.legend(handles, labels, loc=[-0.4, 0.2])
                ax.axis("off")
                continue
            patient = pat_order[pat]
            data = df_noEZ[df_noEZ.patient == patients[patient]]
    
            ax = fig.add_subplot(gs[j, 0])
            x = "kind_seeg"
            order = [r"$PZ_{SEEG}$", "other"]
            np.random.seed(123)
            sns.stripplot(data=data,
                          ax=ax,
                          x=x,
                          palette=["deepskyblue", "lightgrey"],
                          order=order,
                          **stripdic)
            sns.boxplot(data=data,
                        ax=ax,
                        x=x,
                        order=order,
                        **boxdic)
            add_stat_annotation(ax,
                                x=x,
                                box_pairs=[order],
                                data=data,
                                order=order,
                                **statsdic)
            ax.set(ylabel="", xlabel="")
            ax.set_title(patients[patient], x=1, y=.98)
            ax.spines['right'].set_visible(False)
            if i == 0:
                ax.set_ylabel(r"$t_{rec}(s)$")
            ax = fig.add_subplot(gs[j, 1])
            x = "kind_clin"
            order = [r"$PZ_{Clin}$", "other"]
            np.random.seed(123)
            sns.stripplot(data=data,
                          ax=ax,
                          x=x,
                          palette=["darkorange", "w"],
                          order=order,
                          **stripdic)
            sns.boxplot(data=data,
                            ax=ax,
                            x=x,
                            order=order,
                            **boxdic)
            add_stat_annotation(ax,
                                x=x,
                                box_pairs=[order],
                                data=data,
                                order=order,
                                **statsdic)
            ax.set(ylabel="", xlabel="")
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            pat += 1
        space_left += .2
        space_right += .2
    plt.show()
    

def fig25(df):
    plt.rcParams.update(plt.rcParamsDefault)
    fontsize = 17
    plt.rcParams["axes.labelsize"] = "%f" % fontsize
    plt.rcParams["legend.fontsize"] = "%f" % fontsize
    plt.rcParams['ytick.labelsize'] = fontsize
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['xtick.major.size'] = fontsize*.3
    plt.rcParams['xtick.major.width'] = fontsize*.1
    plt.rcParams['ytick.major.size'] = fontsize*.3
    plt.rcParams['ytick.major.width'] = fontsize*.1
    
    x = "short_EZ"
    y = "rec_time"
    meth = "spearman"
    
    spearman_rho = lambda df, pat: df[df.patient==pat].corr(method=meth).loc[x, y]
    spearman_pval = lambda df, pat: df[df.patient==pat].corr(pval_df(meth)).loc[x, y]
    
    df_single_EZ = df.loc[df.num_EZ == 1]
    
    _, ax = plt.subplots(1, 1, figsize=(15, 7))
    
    for pat in df_single_EZ.patient.unique():
        rho = spearman_rho(df_single_EZ, pat)
        pval = spearman_pval(df_single_EZ, pat)
        sns.regplot(x=x,
                    y=y,
                    data=df_single_EZ[df_single_EZ.patient==pat],
                    label=fr"{pat} ($\rho={rho:.2f}$, $p={pval:.1e}$)",
                    scatter_kws={"s": 20, "edgecolor": "k", "linewidth": .3},
                    line_kws={"lw": 2})
    ax.set_xlabel("Shortest path all")
    ax.set_ylabel("$t_{rec}(s)$")
    ax.legend(markerscale=2)
    plt.show()
