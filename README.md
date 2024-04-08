# ComfyUI-MotionThiefExperiment

This is an experimental node pack to test using reference videos for their motion.

It isn't compatible with a lot of things as this is a hacky implementation for experiments only.

## Examples
See example workflow in `example_workflows` to get started. It uses the basic mechanism but there is also an advanced (Custom) settings.

Reference videos are the first on the left. Videos titles are the "motion prompt"

https://github.com/logtd/ComfyUI-MotionThiefExperiment/assets/160989552/396ddddc-b4c2-4e55-a8c8-516981ad688e




https://github.com/logtd/ComfyUI-MotionThiefExperiment/assets/160989552/6ca57165-8517-4d06-bf03-6614e4d971e8




https://github.com/logtd/ComfyUI-MotionThiefExperiment/assets/160989552/ed9f728a-989a-4b6e-bf27-1e82f50fdc8a


## Recipes

You can chain together multiple Custom settings to get different outcomes. Some common combinations are:
* "Normal" can be applied to almost all input/outputs except the last few for the default motion transfer
* "K" and "V" can be applied to high input and low output blocks
* "K" and "Q" can be applied to the last output blocks
* Combining the two KV and KQ above can give a very strong motion transfer
* There are likely other combinations for different results
