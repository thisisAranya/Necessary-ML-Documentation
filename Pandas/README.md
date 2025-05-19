# 🐼 Pandas Complete Reference for Data Analysis and Machine Learning

A comprehensive guide with beginner-friendly code examples, outputs, and descriptions for learning **Pandas**, the most widely used Python library for data manipulation and analysis.

---

## 📚 Table of Contents

- [1. Creating DataFrames and Series](#1-creating-dataframes-and-series)
- [2. Inspecting Data](#2-inspecting-data)
- [3. Indexing and Slicing](#3-indexing-and-slicing)
- [4. Selection with Conditions](#4-selection-with-conditions)
- [5. Data Types and Conversion](#5-data-types-and-conversion)
- [6. Handling Missing Values](#6-handling-missing-values)
- [7. Renaming and Replacing](#7-renaming-and-replacing)
- [8. Sorting and Ranking](#8-sorting-and-ranking)
- [9. Aggregation and GroupBy](#9-aggregation-and-groupby)
- [10. Applying Functions](#10-applying-functions)
- [11. Working with Text Data](#11-working-with-text-data)
- [12. Working with Dates](#12-working-with-dates)
- [13. Merging, Joining, and Concatenating](#13-merging-joining-and-concatenating)
- [14. Pivot Tables and Crosstabs](#14-pivot-tables-and-crosstabs)
- [15. Reading and Writing Files](#15-reading-and-writing-files)
- [16. Duplicate and Unique Values](#16-duplicate-and-unique-values)
- [17. Filtering and Querying](#17-filtering-and-querying)
- [18. MultiIndex and Hierarchical Data](#18-multiindex-and-hierarchical-data)
- [19. Window Functions (Rolling, Expanding)](#19-window-functions-rolling-expanding)
- [20. Plotting with Pandas](#20-plotting-with-pandas)

---

## 1. Creating DataFrames and Series
```python
import pandas as pd

# Series
s = pd.Series([1, 2, 3], name="MySeries")

# DataFrame from dictionary
df = pd.DataFrame({
    "Name": ["Alice", "Bob"],
    "Age": [25, 30]
})

# DataFrame from list of lists
df2 = pd.DataFrame([["Tom", 28], ["Jerry", 26]], columns=["Name", "Age"])
```
**Output:**
```
s:
0    1
1    2
2    3
Name: MySeries, dtype: int64

df:
    Name  Age
0  Alice   25
1    Bob   30

df2:
    Name  Age
0    Tom   28
1  Jerry   26
```

## 2. Inspecting Data
```python
print(df.head())       # First 5 rows
print(df.tail())       # Last 5 rows
print(df.info())       # Summary
print(df.describe())   # Stats summary
print(df.shape)        # (rows, columns)
print(df.columns)      # Column names
print(df.dtypes)       # Data types
```

## 3. Indexing and Slicing
```python
print(df["Name"])        # Single column
print(df[0:1])            # Row slicing
print(df.loc[0, "Name"]) # Label-based access
print(df.iloc[0, 0])      # Position-based access
```

## 4. Selection with Conditions
```python
print(df[df["Age"] > 25])       # Filter rows
print(df[(df.Age > 20) & (df.Name == "Bob")])
```

## 5. Data Types and Conversion
```python
print(df["Age"].astype(float))        # Convert type
print(pd.to_datetime(["2023-01-01", "2023-02-01"]))
```

## 6. Handling Missing Values
```python
df = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, None]})
print(df.isnull())       # Check NaNs
print(df.fillna(0))      # Fill NaNs
print(df.dropna())       # Drop rows with NaNs
```

## 7. Renaming and Replacing
```python
print(df.rename(columns={"A": "Alpha"}))
print(df.replace({1: 100}))
```

## 8. Sorting and Ranking
```python
df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [85, 92]})
print(df.sort_values(by="Score", ascending=False))
print(df["Score"].rank())
```

## 9. Aggregation and GroupBy
```python
df = pd.DataFrame({"Dept": ["IT", "HR", "IT"], "Salary": [100, 200, 300]})
grouped = df.groupby("Dept").mean()
print(grouped)
```

## 10. Applying Functions
```python
print(df["Salary"].apply(lambda x: x * 1.1))
print(df.applymap(lambda x: str(x) + "!"))
```

## 11. Working with Text Data
```python
df = pd.DataFrame({"Names": ["Alice Smith", "Bob Ray"]})
print(df["Names"].str.upper())
print(df["Names"].str.split())
```

## 12. Working with Dates
```python
df = pd.DataFrame({"Date": pd.to_datetime(["2023-01-01", "2023-02-01"])})
print(df["Date"].dt.month)
```

## 13. Merging, Joining, and Concatenating
```python
df1 = pd.DataFrame({"ID": [1, 2], "Name": ["A", "B"]})
df2 = pd.DataFrame({"ID": [1, 2], "Score": [90, 95]})
print(pd.merge(df1, df2, on="ID"))
print(pd.concat([df1, df2], axis=1))
```

## 14. Pivot Tables and Crosstabs
```python
df = pd.DataFrame({"Dept": ["IT", "HR", "IT"], "Gender": ["M", "F", "F"], "Salary": [100, 200, 300]})
print(pd.pivot_table(df, values="Salary", index="Dept", columns="Gender"))
print(pd.crosstab(df["Dept"], df["Gender"]))
```

## 15. Reading and Writing Files
```python
# Reading
pd.read_csv("file.csv")

# Writing
df.to_csv("output.csv", index=False)
```

## 16. Duplicate and Unique Values
```python
df = pd.DataFrame({"A": [1, 2, 2, 3]})
print(df.duplicated())
print(df.drop_duplicates())
print(df["A"].unique())
```

## 17. Filtering and Querying
```python
print(df.query("A > 1"))
```

## 18. MultiIndex and Hierarchical Data
```python
arrays = [["A", "A", "B"], [1, 2, 1]]
index = pd.MultiIndex.from_arrays(arrays, names=("Group", "Sub"))
df = pd.DataFrame({"Data": [100, 200, 300]}, index=index)
print(df)
```

## 19. Window Functions (Rolling, Expanding)
```python
df = pd.DataFrame({"A": [1, 2, 3, 4]})
print(df["A"].rolling(window=2).mean())
print(df["A"].expanding().sum())
```

## 20. Plotting with Pandas
```python
import matplotlib.pyplot as plt

df = pd.DataFrame({"X": [1, 2, 3], "Y": [2, 4, 1]})
df.plot(x="X", y="Y", kind="line")
plt.show()
```

---

This reference covers all foundational operations in Pandas for data analysis and ML workflows. Use this as a cheat sheet or learning resource to master tabular data in Python.
