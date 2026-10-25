# Explanatory icons for Fig. 1

Seven individual icons (donor, islet, beta cell, patch-clamped cell, droplet channel, granules at the membrane, RNA)
were generated once with GPT-Image-2 (OpenAI) on 2026-10-25 from the text prompts listed in `PROMPTS.tsv`, using
`generate_icon.py` (the endpoint and key are read from the environment variables `IMAGE_API_BASE` and `IMAGE_API_KEY`
and are never stored). The PNG outputs were cropped to their content and converted to vector paths with
`vectorize_icons.py` (vtracer), giving `vector/*.svg`.

`scripts/figures/Fig1_framework.py` draws these vector icons through `schematic.Canvas.vector_icon`; every other element
of the figure (layout, panels, diagrams, traces, mini-plots, arrows, boxes and text) is drawn in code. The figure is
therefore entirely vector and rebuilding it does not call the image service. The figure legend and the manuscript's AI
statement state that only these individual icons were generated with AI. They are not used in data figures or in the
graphical abstract.
