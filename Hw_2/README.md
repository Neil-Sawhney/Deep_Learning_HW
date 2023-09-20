# Linear Regression with Basis Expansion
- Setup Virtual Enviroment (if you want).

- Install requirements with ``` pip install -r requirements.txt ```.

- Run ``` python mlp.py ```.
    - Change parameters in the ``` ./config.yaml ``` file.

- View the results in ``` ./artifacts/mlp.pdf ```.

### My attempts at finding a functional form for f
- At first I used no output activation, which caused problems because the probabilities didnt exist most of the time.
- I also had an issue of Nan and Inf loss's untill i added a small constant of 1e-7 to the log.
- I then tried a hard clipping activation, which caused the loss to be 0 very frequently bc of negative values.
- That inspired me to use a sigmoid output activation, which worked well.
- The function worked fine without the L2 regularization, but it seemed to generalize better with it.

# Side Note - apologies for the name of this tar being wrong, I undid the turn in on teams and reuploaded it with the correct name, but i guess it doesn't allow reuploading the same file name so it changes it automatically.