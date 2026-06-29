# FIV demo — VHS tape (charmbracelet/vhs)
# Render:  vhs media/fiv_demo.tape
# Follows DEMO_SCRIPT.local.md. Outputs GIF + MP4.

Output media/fiv_demo.gif
Output media/fiv_demo.mp4

Set Shell bash
Set FontSize 18
Set Width 1280
Set Height 800
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 60ms
Set PlaybackSpeed 1.0

# --- 0:00 cold terminal ---
Hide
Type "cd /home/swarm/rinadelph/cerebras-hackathon"  Enter
Type "clear"  Enter
Show
Sleep 800ms

# --- Shot 1: the setup — run the framing-reveal demo (paced) ---
Type "python3 scripts/demo.py --id tf-test-11 --slow 1.0"
Sleep 700ms
Enter

# --- Shot 2+3: the reveal + headline play out under --slow pacing ---
# (table prints, then 8 framings reveal one-by-one green->red, then the
#  powered headline panel: 0.320 -> 0.184, p=9.9e-9, -43%)
Sleep 11s

# Let the final headline + 43% line breathe on screen
Sleep 3500ms

# --- Shot 4: close card ---
Type "clear"  Enter
Sleep 400ms
Type "echo 'The question is the verdict.  We measured it. We cut wrong-flips 43%.'"  Enter
Sleep 600ms
Type "echo 'github.com/Swarm-Code/framing-induced-variance   @CerebrasSystems  @googlegemma'"  Enter
Sleep 3s
