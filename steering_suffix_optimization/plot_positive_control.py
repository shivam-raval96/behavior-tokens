"""Plot the layer-10 direct-steering causal gate."""
from pathlib import Path
import argparse,json
import matplotlib.pyplot as plt

def plot(run_dir: Path) -> Path:
    rows=json.loads((run_dir/"results.json").read_text())["curve"]
    labels=[f"{x['additive_norm']:+.3f}" for x in rows]
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    bars=axes[0].bar(labels,[x["asr"] for x in rows],color=["#2a9d6f","#7b8794","#c4554d"])
    axes[0].set(xlabel="Signed additive norm",ylabel="HarmBench ASR",ylim=(0,1),title="Layer-10 causal sign control")
    axes[0].grid(axis="y",alpha=.22)
    for bar,row in zip(bars,rows):
        axes[0].text(bar.get_x()+bar.get_width()/2,row["asr"]+.025,f"{row['asr']:.0%}",ha="center")
    axes[1].plot(labels,[x["eos_rate"] for x in rows],marker="o",label="EOS rate")
    axes[1].plot(labels,[x["mean_repeated_trigram_fraction"] for x in rows],marker="o",label="Repeated trigram fraction")
    axes[1].set(xlabel="Signed additive norm",ylabel="Rate",ylim=(-.03,1.03),title="Generation quality diagnostics")
    axes[1].grid(alpha=.22); axes[1].legend(frameon=False)
    fig.suptitle("Validated layer-10 refusal direction: direct intervention")
    output=run_dir/"direct_steering_control.png"; fig.savefig(output,dpi=180); plt.close(fig); return output

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("run_dir",type=Path); args=parser.parse_args(); print(plot(args.run_dir))
if __name__=="__main__": main()
