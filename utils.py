import pandas as pd

def csv2seq(foldername, filename, note):
    with open(f"{foldername}/{filename}", "r") as f:
        line = [line.strip("\n").split(", ") for line in f if len(line.split(", "))==6]

    division = int(line[0][-1])
    scale = 1024/division # Normalize to 1024 division

    df = pd.DataFrame(line, columns=["track", "time", "tipe", "channel", "note", "velocity"])
    df = df.loc[df.tipe.isin(["Note_on_c", "Note_off_c"])]

    df.time = df.time.apply(lambda x: round(int(x)*scale))
    df.track = df.track.apply(int)
    df.note = df.note.apply(lambda x: midi2note[int(x)])
    df.velocity = df.velocity.apply(int)

    # Note_on_c with 0 velocity means Note_off_c
    df.tipe[(df.tipe=="Note_on_c") & (df.velocity==0)] = "Note_off_c"
    
    df.drop(["channel", "velocity"], axis=1, inplace=True)

    filename = filename.strip(".csv")
    note[filename] = {}
    for track in df.track.unique():
        df_on = df.loc[(df.tipe=="Note_on_c") & (df.track==track)]
        df_off = df.loc[(df.tipe=="Note_off_c") & (df.track==track)]
        df_on.durr = [df_off[(df_off.note==note) & (df_off.time > time)].iloc[0, 1] for time, note in zip(df_on.time.values, df_on.note.values)] - df_on.time
        df_on["next"] = df_on.time.diff().shift(-1).fillna(0)
        df_on.note = df_on.note + "-" + df_on.durr.apply(lambda x: str(int(x))) + "-" + df_on.next.apply(lambda x: str(int(x)))
        note[filename][track-1] = " ".join(df_on.note.values)    
    return note    