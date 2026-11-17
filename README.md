# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Arik2779/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Arik2779/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for Grand Theft Auto VI's launch-day Twitch viewers, hour by hour. Given the first hour's viewer and channel counts plus the release-day curves of 38 past big launches, it predicts hours 2–24 as five quantiles each (5%, 25%, 50%, 75%, 95%) on the log scale. Per hour it fits a weighted regression of each comparable launch's log-ratio to hour 1 on the launch's own hour-1 size (viewers and channels) plus a rolling-midnight dummy — rolling launches unlock region by region, jump at hour 5 when the US East coast unlocks, and run about 0.6 log units higher all day than simultaneous-global launches at the same hour-1 size — with rolling curves weighted double, the case's own unlock class read from its notes, and the prediction shrunk 35% toward the rolling launches' median curve; spread comes from the weighted residual dispersion. Practice pinball loss 0.152 vs the house baseline's 0.387 (v1: 0.188). Python 3.13, standard library only, deterministic, under a second per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=808mq1jq&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
