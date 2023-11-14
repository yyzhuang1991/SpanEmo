# 1. move test.json somewhere in this folder.
run the following command: 
python scripts/test_emotion_reaction.py --max-length 128 --test-path my_test_data/test.json --model-path joint.pt

python scripts/test_emotion_reaction_using_events.py --test-path my_test_data/test.json.final --model-path joint.pt