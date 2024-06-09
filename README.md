
# Gesture Video Player Controls
In 2023 my former classmate oZep created an ASL reader using OpenCV, that's
really cool and probably has all sorts of implications for people who know
and/or need ASL but I'm here to solve a more pressing and important matter.<br><br>

Sometimes when I'm eating food and watching cartoons I can't hit the space bar
without getting my keyboard dirty. (see *figure 1* below)<br><br>
![img](/imgs/figure1.png)

I think it should be obvious to anyone that my problems are much more severe and
deserve way more attention than people with small inconveniences such as not
being able to speak or hear other people speak.<br><br>

Yet somehow, our unjust society has allowed my plight to continue unaided. Until
now. 


## Configuration
At the top of the file there are two dictionaries `gesture_commands` and
`hand_gestures`. The first maps the name of a gesture (a simple string) to a
function it should run. In my case, they are all `lambda` wrapped calls to
`xdotool` commands.<br><br>
The second dictionary is `hand_gestures` which maps the name of a gesture to a
21x3 dimensional array specifying the location of each hand bone. That is of
course very annoying to specify, so you can launch the program, make the
appropriate gesture, and press the space bar to print out the vector that
represents it.<br><br>

By default there are only two hand gestures (a peace sign and a thumbs up
rotated 90 degrees to the left). I have them bound to press the mouse and press
the tab button respectively.<br><br>

If you would like to change which device it uses you can consider crying in a
corner or maybe unplugging some web cams.


## How does it actually work?
The ~~stolen~~ forked code uses OpenCV to detect the location of each hand
segment. Each segment contains a tuple (x, y, z) specifying its supposed
location. Those locations are in cartesian coordinates relative to the camera,
so they are first re-centered to the middle of the hand. Then the euclidean
distance is taken between the current re-centered hand and the re-centred hand
stored in `hand_gestures`. If the distance is less than the tolerance, and its
been at least 2 seconds since we last saw this gesture, we trigger the relevant
function in `gesture_commands`.

## Installation Info
This was built and tested with `OpenCV=4.10.0` and `mediapip=0.10.14` so those
are probably the best versions to use. If `venv` is a hassle then I'll put a
compile version (for Linux) in the releases section- but since its configured
via source files it'd be hard to customize the gestures.
