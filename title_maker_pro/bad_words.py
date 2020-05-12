# derived-from grawlix-racism
# WARNING: THIS PLUGIN DEALS DIRECTLY WITH TERMS THAT MANY FIND OFFENSIVE, 
# IMPOLITE, AND/OR DISTASTEFUL. AS SUCH, ALL SOURCE CODE SHOULD BE CONSIDERED 
# NSFW/NOT SAFE FOR WORK UNLESS OTHERWISE STATED. PLEASE READ THESE FILES AT 
# YOUR OWN RISK!

import re

patterns = [
    r"n+[i1]+gg+e+r+[s\$z]+",
    r"(\b|^|[^s])n+[\W02-9_]{0,42}[i1]+[\W02-9_]{0,42}g[\W\d_]{0,42}g+[\W0-24-9_]{0,42}[e3]+[\W0-24-9_]{0,42}r+",
    r"m+[i1]+gg+[e3]+r+",
    r"yV+[i1]+gg+[e3]+r+",
    r"d+[a@]+r+k+[i1]+[e3]+s+",
    r"d+[\W\d_]{0,42}[a@]+[\W\d_]{0,42}r+[\W\d_]{0,42}k+[\W02-9_]{0,42}[i1]+[\W024-9_]{0,42}[e3]+",
    r"/[s$z]+h+v+[a@]+t+[s$z]+[a@]+[s$z]+",
    r"/[s$z]+h+v+[a@]+t+[s$z]+[a@]+",
    r"/(\b|^)[s$]+p+[i1]+c+k{0,32}[s$]+",
    r"/(\b|^)[s$]+[\W\d_]{0,42}p+[\W02-9_]{0,42}[i1]+[\W02-9_]{0,42}c+(\b|$)",
    r"/w+[e3]+t+b+[a@]+c+k+[s$z]+",
    r"/w+[e3]+t+b+[a@]+c+k+",
    r"/k+[i1]+k+[e3]+[s$z]+",
    r"/k+[\W02-9_]{0,42}[i1]+[\W02-9_]{0,42}k+[\W0124-9_]{0,42}[e3]+(\b|$)",
    r"/g+[\W1-9_]{0,42}[o0][\W1-9_]{0,42}[o0]+[\W1-9_]{0,42}k+[\W\d_]{0,42}[s$]+",
    r"/(\b|^)g+[\W1-9_]{0,42}[o0][\W1-9_]{0,42}[o0]+[\W1-9_]{0,42}k+(?!y)",
    r"/r+[a@]+g+[\W\d_]{0,42}h+[e3]+[a@]+d+[s$]+",
    r"/r+[\W\d_]{0,42}[a@]+[\W\d_]{0,42}g+[\W\d_]{0,42}h+[\W0-24-9_]{0,42}[e3]+[\W0-24-9_]{0,42}[a@]+[\W\d_]{0,42}d+",
    r"/t+[o0]+w+[e3]+[l1]+[\W02-9_]{0,42}h+[e3]+[a@]+d+[s$]+",
    r"/t+[\W1-9_]{0,42}[o0]+[\W1-9_]{0,42}w+[\W0-24-9_]{0,42}[e3]+[\W024-9_]{0,42}[l1]+[\W02-9_]{0,42}h+[\W0-24-9_]{0,42}[e3]+[\W0-24-9_]{0,42}[a@]+[\W\d_]{0,42}d+",
    r"/[i1]+n+j+u+n+[s$]+",
    r"/[i1]+[\W02-9_]{0,42}n+[\W\d_]{0,42}j+[\W\d_]{0,42}u+[\W\d_]{0,42}n+(\b|$)",
    r"/(\b|^)[s$]+q+u+[a@]+w+s+",
    r"/(\b|^)[s$]+q+u+[a@]+w+(\b|$)",
    r"/g[o0][l1][l1][i1y]w[o0]g+[s$]",
    r"/g[o0][l1][l1][i1y]w[o0]g+",
    r"/w+[\W1-9_]{0,42}[o0]+[\W1-9_]{0,42}g+[\W\d_]{0,42}[s$]+",
    r"/(\b|^)w+[\W1-9_]{0,42}[o0]+[\W1-9_]{0,42}g+(\b|$)",
    r"/[ck][a@]ff[ie3]r[s$z]",
    r"/[ck][a@]ffr[e3][s$z]",
    r"/[ck][\W\d_]{0,42}[a@][\W\d_]{0,42}f[\W\d_]{0,42}f[\W\d_]{0,42}r[\W0-24-9_]{0,42}[e3]",
    r"/[ck]+[\W\d_]{0,42}[a@]+[\W\d_]{0,42}(?:f[\W024-9_]{0,42})+[i1e3]+[\W024-9_]{0,42}r+",
    r"/[s$]h[i1]t[s$]k[i1]n[s$]",
    r"/[s$]h[i1]t[s$]k[i1]n",
    r"/[l1][a@]tr[i1]n[o0][s$z]",
    r"/[l1][a@]tr[i1]n[o0]",
    r"/ch[i1]nk[e3]rb[e3][l1][l1]+",
    r"/[s$]hv[o0][o0]g[a@][s$]",
    r"/[s$]hv[o0][o0]g[a@]",
    r"/n[e3]gr[e3][s$z][s$z]+",
    r"/[s$][a@]mb[o0][e3]*[s$]",
    r"/[s$][\W\d_]{0,42}[a@][\W\d_]{0,42}m[\W\d_]{0,42}b[\W1-9_]{0,42}[o0]",
]

ULTRA_BAD_REGEX = re.compile(
    f'({"|".join(p for p in patterns)})',
    flags=re.IGNORECASE
)


def grawlix(s):
    return ULTRA_BAD_REGEX.sub(lambda m: '#!$%!#&@%!$@#$@!$'[:len(m.group())], s)