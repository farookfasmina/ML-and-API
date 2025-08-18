from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

out = Path("data")
out.mkdir(exist_ok=True, parents=True)

iris = load_iris(as_frame=True)
X = iris.data.rename(columns={
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)":  "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)":  "petal_width",
})
y = pd.Series(iris.target, name="species_id")
species_name = pd.Series([iris.target_names[i] for i in iris.target], name="species")

df = pd.concat([X, species_name, species_id], axis=1) if (species_id:=y) is not None else None
df = pd.concat([X, species_name, species_id], axis=1)
df.to_csv(out / "iris.csv", index=False)
print("Saved", out / "iris.csv")
