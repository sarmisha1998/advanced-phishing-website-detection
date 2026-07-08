import pandas as pd


def create_features(df):

    print("Starting Feature Engineering...")

    # URL Length
    df["URL_Length"] = df["URL"].apply(len)

    # Number of dots
    df["Dot_Count"] = df["URL"].str.count(r"\.")

    # Number of slashes
    df["Slash_Count"] = df["URL"].str.count("/")

    # Number of hyphens
    df["Hyphen_Count"] = df["URL"].str.count("-")

    # Number of digits
    df["Digit_Count"] = df["URL"].str.count(r"\d")

    # Contains HTTPS
    df["HasHTTPS"] = df["URL"].str.startswith("https").astype(int)

    print("Feature Engineering Completed.")

    return df