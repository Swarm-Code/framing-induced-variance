# FIV cinematic demo — VHS tape
# Render:  vhs media/fiv_cinematic.tape

Output media/fiv_cinematic.gif
Output media/fiv_cinematic.mp4

Set Shell bash
Set FontSize 17
Set Width 1400
Set Height 860
Set Padding 30
Set Theme "Catppuccin Mocha"
Set TypingSpeed 50ms

Hide
Type "cd /home/swarm/rinadelph/cerebras-hackathon && clear"  Enter
Type "python3 scripts/demo_cinematic.py --id tf-test-11 --speed 1.0"  Enter
Sleep 800ms
Show

# full animated reveal: title -> method -> table -> 8 framings -> noise ->
# asymmetry -> robustness -> speed -> headline -> close (~46s of animation)
Sleep 49s
