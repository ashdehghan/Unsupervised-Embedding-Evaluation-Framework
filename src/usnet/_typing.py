"""Type aliases and protocols for usnet."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import networkx as nx
    import pandas as pd

# Array types
ArrayLike: TypeAlias = npt.NDArray[np.floating]
IntArrayLike: TypeAlias = npt.NDArray[np.integer]

# Graph types
Graph: TypeAlias = "nx.Graph"

# DataFrame types
DataFrame: TypeAlias = "pd.DataFrame"

# Feature names
FeatureName: TypeAlias = str

# Node identifier
NodeId: TypeAlias = int | str
