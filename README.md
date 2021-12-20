# Installation

Before starting make sure you have `python3` installed. This code has been developed and tested on `python 3.8.6`. The recommended approach to installing dependencies:

    Dependencies can be install via the following command:
    pip3 install -r requirements.txt

After successful installation you should be able to perform social behavior analysis on pose files. As an example you might run the command like:

python -u gensocialstats.py \
   --social-config social-config.yaml \
   --batch-file poses.txt \
   --root-dir /full/path/to/poses/ \
   --out-file results.yaml

# Licensing

This code is released under MIT license.
