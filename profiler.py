import cProfile
from HPO import main

cProfile.run('main(ATTN_ENCODE_OPT.json)', 'profile_output.txt')