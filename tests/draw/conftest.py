"""Setup shared by the draw tests."""

import matplotlib

# The pyplot figures of these tests need no window: Agg, as CI sets it
# with MPLBACKEND (a desktop backend fails without a display or Tk).
matplotlib.use("Agg")
