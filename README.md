# make_swe_bench_verified_mini

Link to the dataset on huggingface: [https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini)

If you use the [Inspect implementation](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/swe_bench), you can merely switch the `dataset: str = "princeton-nlp/SWE-bench_Verified",` to `dataset: str = "MariusHobbhahn/swe-bench-verified-mini",` in the "swe_bench.py" file.

SWEBench-verified-mini is a subset of SWEBench-verified that uses 50 instead of 500 datapoints, requires 5GB instead of 130GB of storage (only using django and sphinx environments) and has approximately the same distribution of performance, test pass rates and difficulty as the original dataset on 16 different models.

We compare between four datasets:
- **Full**: all 500 datapoints of SWEBench-verified
- **K-mean representative**: 50 datapoints selected by k-means clustering (but no size optimization)
- **Random**: 50 random datapoints
- **Size optimized**: 50 datapoints selected by Linear Programming to minimize storage size while use k-means to keep the distribution of scores (performance, test pass rates, difficulty) similar to the full dataset.

We want the numbers of the subsets to be as close as possible to the full dataset, i.e. the orange, green and especially the red bar should be as close as possible to the blue bar.

![Mean comparison](figures/means_comparison.png)
![Difficulty distributions](figures/difficulty_distributions.png)

## Context

[SWEBench-verified](https://www.swebench.com/) is a great dataset, but it is very large. Thus, I wanted to create a smaller version that is a good proxy for the overall performance of models on the dataset. I call this dataset SWEBench-verified-mini.

SWEBench-verified-mini is a subset of SWEBench-verified that uses 50 instead of 500 datapoints, requires 5GB instead of 130GB of storage and has approximately the same distribution of performance, test pass rates and difficulty as the original dataset (using 16 models for comparison).

Note that I'm primarily combining work from other people, so most of the credit should go to them:
- [Jiminez et al.](https://arxiv.org/abs/2310.06770) for creating the original dataset.
- [The OpenAI evals contractor team](https://openai.com/index/introducing-swe-bench-verified/) for taking the dataset and creating SWEBench-verified. 
- My MATS scholars [Govind](https://pimpale.github.io/) and [Axel](https://www.linkedin.com/in/axelhojmark/) for running the full dataset on a decent number of models for a different project so that I can compare the performance of models on the full dataset.
- [TinyBenchmarks by Polo et al.](https://arxiv.org/abs/2402.14992) for good ideas on how to build tiny versions of bigger benchmarks. I use a different approach than them but was inspired by their work but used their paper for inspiration. Because I use a different approach, I decided to call my dataset 'mini' instead of 'tiny'.

## How to make SWEBench-verified-mini

We want to select 50 datapoints of the 500 datapoints in SWEBench-verified such that:
1. The marginal distribution of important scores (performance, test pass rates, difficulty) are similar to the full dataset. 
2. The overall storage sizes of the dataset is minimized. SWEBench-verified requires 130GB or 260GB of storage (depending on whether you want to create both arm64 and x86_64 versions of the docker environments). This is because you need to create 40 different docker environments. We want to find a subset of SWEBench-verified that is a good proxy while using as few docker environments as possible to minimize storage size.

We first run k-means clustering on all datapoints. We then use Linear Programming to select the 50 datapoints such that the storage size is minimized while keeping the distribution of scores (performance, test pass rates, difficulty) similar to the full dataset my making sure that the k-meansclusters are proportional to the full dataset.  

## How to run the code

If you want to run the code, you can either run the file `run_all.py` or run the individual files. We provide the `.eval` files of 16 model runs on the full SWEBench-verified dataset to the `logs` folder. If you want to run this for your own evals, you can add your own `.eval` files to the `logs` folder. We have not tried very hard to elicit maximal performance from the models and instead focused on comparability between runs.

If you want to run the individual files, you have to start with `get_docker_image_sizes.py` and `extract_data_from_logs.py`. These are data wrangling scripts.

Then you can run `add_metadata_to_data.py` and `generate_subsets.py`.

If you want to compare the generated subsets, you can run `compare_subsets.py`.

Finally, you can run `make_new_huggingface_dataset.py` to create the new dataset.

Note that the code is fairly hacky and has not been optimized for usability.

## Notes on running SWEBench

I found installing SWEBench on my Mac to be a pain. For some reason, the installation keeps hanging constantly and I needed to manually restart it a lot of the time. It also takes 130GB (or 260GB if you install environments for both arm64 and x86_64) of storage.

If you want to install the full SWEBench, you should go into your Docker Desktop settings to Resources and increase memory limit, swap and virtual disk limit. In the General tab, you might also want to go to Virtual Machine Options, select Apple Virtualization Framework and tick "Use Rosetta for x86_64/amd64 emulation on Apple Silicon". 

Finally, I went into the file `/Users/<user>/miniconda3/envs/<env>/lib/python3.12/site-packages/swebench/harness/test_spec.py` and hardcoded the selection because I think the code decided to install both arm64 and x86_64 environments. 

```
# remove platform selection as quick fix
#    if platform.machine() in {"aarch64", "arm64"}:
#        # use arm64 unless explicitly specified
#        arch = "arm64" if instance_id not in USE_X86 else "x86_64"
#    else:
#        arch = "x86_64"
# and replace with always selecting x86_64 or arm64
    arch = "x86_64"  # Always use x86_64 regardless of platform; I chose x86_64 because that's recommended in the official repo despite having an M2 chip.
#    arch = "arm64"  # Always use arm64 regardless of platform
```

I have not investigated this in a lot of detail, so it's possible that I'm making things more complicated than they need to be.
