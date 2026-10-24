"""Plot saved paired changes against the independent compact control."""
import argparse
import csv
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
DEST=HERE.parents[1]/"docs/figures"
DATASETS=("concrete_slump","parkinsons","air_quality")
METHODS=("joint","independent_penalty","hybrid_penalty","hybrid_compact")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-scripts",type=Path)
    args=parser.parse_args()
    with (HERE/"analysis/paired.csv").open(encoding="utf-8",newline="") as stream:
        rows=list(csv.DictReader(stream))
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
        "font.size":7,"axes.labelsize":7,"xtick.labelsize":7,"ytick.labelsize":7,
        "svg.fonttype":"none","pdf.fonttype":42,"axes.spines.top":False,
        "axes.spines.right":False,"axes.linewidth":.65})
    fig,axes=plt.subplots(2,3,figsize=(7.480315,5.196850))  # 190 x 132 mm
    fig.subplots_adjust(left=.10,right=.985,bottom=.15,top=.81,wspace=.34,hspace=.60)
    fig.suptitle("Shared/private search: paired accuracy and model size",y=.98,fontsize=10)
    fig.text(.54,.905,"Reference: independent search with compact validation selection",ha="center",fontsize=8)
    colors=("#0072B2","#777777","#D55E00","#009E73")
    titles=("Concrete slump","Parkinson","Air Quality")
    fields=("delta_outer_nrmse","delta_complexity")
    for col,dataset in enumerate(DATASETS):
        for row,field in enumerate(fields):
            ax=axes[row,col]
            ax.set_title(f'{"abcdef"[row*3+col]}   {titles[col]}',loc="left",fontsize=8,pad=8)
            ax.axhline(0,color="#888888",ls=":",lw=.7)
            ax.grid(axis="y",color="#dddddd",lw=.4)
            ax.set_axisbelow(True)
            for x,(method,color) in enumerate(zip(METHODS,colors)):
                part=[r for r in rows if (r["dataset"],r["left"],r["right"])==(dataset,method,"independent_compact")]
                assert len(part)==15
                if not all(r["status"]=="ok" for r in part):
                    ax.text(x,.95,"incomplete",transform=ax.get_xaxis_transform(),rotation=90,rotation_mode="anchor",ha="center",va="top",fontsize=7)
                    continue
                values=[float(r[field]) for r in part]
                ax.scatter(x+np.linspace(-.16,.16,15),values,s=9,color=color,alpha=.45,linewidths=0,zorder=2)
                ax.errorbar(x,statistics.mean(values),yerr=statistics.pstdev(values),fmt="D",markersize=3,
                    color="black",elinewidth=.8,capsize=2,zorder=3)
            ax.set_xlim(-.45,3.45)
            ax.set_xticks(range(4),["Joint","Indep.\npenalty","Hybrid\npenalty","Hybrid\ncompact"])
    axes[0,0].set_ylabel("Outer-fold NRMSE difference",fontsize=7)
    axes[1,0].set_ylabel("Shared-DAG complexity difference",fontsize=7)
    fig.text(.54,.045,"All 15 paired cells; black diamonds: mean +/- descriptive SD. Negative favors the named method.",ha="center",fontsize=7)
    fig.canvas.draw()
    DEST.mkdir(exist_ok=True)
    stem=DEST/"shared_private_paired"
    if args.qa_scripts:
        sys.path.insert(0,str(args.qa_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig,json_out=f"{stem}.alignment.json",overlay_svg=f"{stem}.alignment.svg",
            tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png",dpi=300)
    plt.close(fig)
    print("Exported shared/private paired figure; alignment "+("checked." if args.qa_scripts else "not checked."))


if __name__=="__main__":
    main()
