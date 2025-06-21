# Info

## Explanation

### ProppantPerFoot

1. Hydraulic fracturing creates cracks (fractures) in low-permeability rock (like shale).

2. After the high-pressure fluid is removed, the fractures would normally close back up.

3. Proppants stay behind in the fractures and physically hold them open.

4. This allows oil or gas to flow through the fractures and into the wellbore.

## Duplicates

No duplicates

## Missing Values

### Raw

| Feature            | Missing Values |
| ------------------ | -------------- |
| WellID             | 0              |
| BVHH               | 1641           |
| FormationAlias     | 0              |
| NioGOR             | 497            |
| CodGOR             | 1668           |
| LateralLength      | 0              |
| ProppantPerFoot    | 232            |
| FluidPerFoot       | 264            |
| LeftDistance       | 2448           |
| LeftNeighbourType  | 0              |
| RightDistance      | 2428           |
| RightNeighbourType | 0              |
| TVD                | 313            |
| NormalizedOilEUR   | 0              |
| NormalizedGasEUR   | 0              |

### Positive Only

| Feature            | Missing Values |
| ------------------ | -------------- |
| WellID             | 0              |
| BVHH               | 1635           |
| FormationAlias     | 0              |
| NioGOR             | 493            |
| CodGOR             | 1661           |
| LateralLength      | 0              |
| ProppantPerFoot    | 231            |
| FluidPerFoot       | 255            |
| LeftDistance       | 2424           |
| LeftNeighbourType  | 0              |
| RightDistance      | 2407           |
| RightNeighbourType | 0              |
| TVD                | 313            |
| NormalizedOilEUR   | 0              |
| NormalizedGasEUR   | 0              |

### No Left or Right Neighbors

| Feature            | Missing (Before) | Missing (After) | Change (%) |
| ------------------ | ---------------- | --------------- | ---------- |
| WellID             | 0                | 0               | 0.00%      |
| BVHH               | 1635             | 872             | 53.33%     |
| FormationAlias     | 0                | 0               | 0.00%      |
| NioGOR             | 493              | 491             | 99.6%      |
| CodGOR             | 1661             | 890             | 53.58%     |
| LateralLength      | 0                | 0               | 0.00%      |
| ProppantPerFoot    | 231              | 154             | 66.67%     |
| FluidPerFoot       | 255              | 165             | 64,70%     |
| LeftDistance       | 2424             | 2424            | 100.00%    |
| LeftNeighbourType  | 0                | 0               | 0.00%      |
| RightDistance      | 2407             | 2407            | 100.00%    |
| RightNeighbourType | 0                | 0               | 0.00%      |
| TVD                | 313              | 313             | 100.00%    |
| NormalizedOilEUR   | 0                | 0               | 0.00%      |
| NormalizedGasEUR   | 0                | 0               | 0.00%      |

## Categorical Data

**RightDistance** is NaN if and only if `RightNeighbourType="NoNeighbour"` <br>
**LeftDistance** is NaN if and only if `LeftNeighbourType="NoNeighbour"`

### FormationAlias

'NIOBRARA': 6917, 'CODELL': 2289

### NeighbourType

LeftNeighbourType: `{'Codeveloped': 6022, 'NoNeighbour': 2424, 'Parent': 712})` <br>
RightNeighbourType: `{'Codeveloped': 6030, 'NoNeighbour': 2407, 'Parent': 721})`

### What to do

one-hot vector since only 2 and 3 categories

# Relationships

## Pearson Correlation

# Preprocess

## Removed Non Positive Values

**BVHH**: Is explicitly stated to be positive <br>
**ProppantPerFoot**: The amount (in pounds) of proppant (Amount of pounds cannot be in negative?) <br>
**NormalizedOilEUR**: The amount of oil (in bbl/ft) produced by the well in its lifetime normalized by its lateral length (Amount of oil cannot be negative?) <br>
**NormalizedGasEUR**: The amount of gas (in mcf/ft) produced by the well in its lifetime normalized by its lateral length (Amount of gas cannot be negative?) <br>

```py
temp = data[~((data["BVHH"]<=0) | (data["ProppantPerFoot"]<=0) | (data["NormalizedGasEUR"]<=0) | (data["NormalizedOilEUR"]<=0))]
```

Non positive **BVHH**: 2 <br>
Non positive **ProppantPerFoot**: 42 <br>
Non positive **NormalizedOilEUR**: 1 <br>
Non positive **NormalizedGasEUR**: 3

## Outliers

We calculate the outlier samples after remove non positive values

### TVD

`< 4500`: 1 sample <br>
`> 9500`: 1 sample

### ProppantPerFoot

`>5600`: 4 samples <br>

### FluidPerFoot

`>7100`: 9 samples

### NioGOR

??
Momken 50000:73

### NormalizedOilEUR

<60: 17 samples ??

# Steps

Removed the onehotencode of FormationAlias_NIOBRARA since it is the exact opposite of FormationAlias_CODELL
