(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("exam" "a4paper" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("fontenc" "T1") ("xcolor" "dvipsnames") ("algorithm2e" "linesnumbered" "ruled" "vlined")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "exam"
    "exam11"
    "fontenc"
    "titling"
    "url"
    "amsmath"
    "amsthm"
    "amssymb"
    "graphicx"
    "graphics"
    "listings"
    "color"
    "xcolor"
    "tabularx"
    "ragged2e"
    "courier"
    "textcomp"
    "circuitikz"
    "tikz"
    "karnaugh-map"
    "bytefield"
    "mathrsfs"
    "cancel"
    "algorithm2e"
    "hyperref"
    "my_styles"
    "enumitem"
    "environ")
   (TeX-add-symbols
    '("tab" ["argument"] 0)
    '("subtitle" 1)
    '("ztransform" 1)
    '("fourier" 1)
    '("laplace" 1)
    '("invlaplace" 1)
    "myvspace")
   (LaTeX-add-labels
    "fig:bandpass"
    "fig:time_data"
    "fig:bandpass_regions"
    "fig:define_freqs"
    "app_sub_FIR_create")
   (LaTeX-add-environments
    '("eqnsection" 2)))
 :latex)

