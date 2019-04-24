import os
import sys
import numpy as np
from matplotlib import pyplot as pl

cols = ['r', 'g', 'b']
centers = [10, 20, 30]
taus = [-5, -15, -5]

tau = -10  # transition time, use this to find the best b that gives slope a*b/4
a = 1.2
# c = 20
d = 4.5
fSig = lambda x,c: a + d / (1 + np.exp((-4 / tau) * (x - c)))
fSigTau = lambda x, c, t: a + d / (1 + np.exp((-4 / t) * (x - c)))


xMin = 0
xMax = 40

xOuterDelta = 0
xs = np.linspace(xMin - xOuterDelta, xMax + xOuterDelta, 50)

######## 3 separate diagrams with one sigmoid each ########
# for g in range(3):
#   fig = pl.figure(1)
#   pl.plot(xs, fSig(xs, c=centers[g]), color=cols[g], lw=15)
#   pl.show()
#   fig.savefig('sig%d.png' % g)
#   # print(adsad)

######## make diagram with all 3 sigs ########
# fig = pl.figure(1)
# for g in range(3):
#   pl.plot(xs, fSig(xs, c=centers[g]), color=cols[g], lw=8)
# pl.xlabel('disease stage', fontsize=18)
# pl.ylabel('biomarker value', fontsize=18)
# ax = pl.gca()
# ax.set_xticklabels([''])
# ax.set_yticklabels([''])
# fig.show()
# fig.savefig('sigAll.png')

####### diagram with one sigmoid but with confidence interval ########
# g = 1
# fig = pl.figure(1)
# pl.plot(xs, fSig(xs, c=centers[g]), color=cols[g], lw=8)
# sigma_pred = 0.5
# pl.fill(np.concatenate([xs, xs[::-1]]),
#           np.concatenate([fSig(xs, c=centers[g]) - 1.9600 * sigma_pred,
#                          (fSig(xs, c=centers[g]) + 1.9600 * sigma_pred)[::-1]]),
#           alpha=.5, fc=cols[g], ec='None')
#
# pl.xlabel('disease stage', fontsize=18)
# pl.ylabel('biomarker value', fontsize=18)
# ax = pl.gca()
# ax.set_xticklabels([''])
# ax.set_yticklabels([''])
# fig.show()
# fig.savefig('sigOneConfidenceInt.png')

######## diagram with biomarkers as step functions #########

fig = pl.figure(1)

lw=8
yvals = [0,0,1,1]
pl.plot([0, 1, 1, 4], yvals,linewidth=lw)
pl.plot([0, 2, 2, 4], yvals,linewidth=lw)
pl.plot([0, 3, 3, 4], yvals,linewidth=lw)

pl.xticks([])
pl.yticks([0,1])

fs = 30
pl.xlabel('disease stage', fontsize=fs)
pl.ylabel('biomarker value', fontsize=fs)
ax = pl.gca()
ax.set_xticklabels([''])
ax.set_yticklabels(['normal', 'abnormal'],fontsize=fs)
ax.set_yticklabels([],fontsize=fs)

fig.show()
fig.savefig('biomkStepFunctions.png')
print(asdas)


######## diagram with many sigmoids clustered into 3 colors ########
np.random.seed(1)
fig = pl.figure(1)
nrClust = 3
nrBiomk = 100
clustAssignTrueB = np.array(np.floor(nrClust * np.random.rand(nrBiomk)), int)
thetasTrue = np.concatenate([np.array(centers).reshape(1,-1),
  np.array(taus).reshape(1,-1)],axis=0).T
covPerturbed = np.zeros((nrClust, 2,2))
noiseCenter = 10
noiseTau = 3
covPerturbed[0,:,:] = np.diag([noiseCenter,noiseTau])
covPerturbed[1,:,:] = np.diag([noiseCenter,4*noiseTau])
covPerturbed[2,:,:] = np.diag([noiseCenter,noiseTau])
print('thetasTrue', thetasTrue)
print('thetasTrue.shape', thetasTrue.shape)
paramsPert = np.array([np.random.multivariate_normal(
  thetasTrue[clustAssignTrueB[b], :], covPerturbed[clustAssignTrueB[b],:,:])
  for b in range(nrBiomk)])

for b in range(nrBiomk):
  pl.plot(xs, fSigTau(xs, paramsPert[b,0], paramsPert[b,1]), '-', color=cols[clustAssignTrueB[b]],
    lw=2, alpha=0.4)

for c in range(3):
  pl.plot(xs, fSigTau(xs, c=centers[c], t=taus[c]), color=cols[c], lw=7)

pl.xlabel('disease stage', fontsize=18)
pl.ylabel('biomarker value', fontsize=18)
ax = pl.gca()
ax.set_xticklabels([''])
ax.set_yticklabels([''])
fig.show()
fig.savefig('sigManyBiomkClustering.png')
print(adsad)


text = r'''
\documentclass[10pt,xcolor=table]{beamer}

\usepackage{graphicx}
\usepackage{caption}
\usepackage{subcaption}

% \usepackage[utf8]{inputenc}
% \usepackage[T1]{fontenc}
\usepackage[table]{xcolor}    % loads also »colortbl«
%  \usepackage{enumitem}
\usepackage{ucltemplate}
\usepackage{color}

\usepackage{pgfgantt} % for grantt charts
\usepackage{rotating}
\usepackage[graphicx]{realboxes}

\usepackage{tikz}
\usetikzlibrary{arrows,positioning, shapes.symbols,shapes.callouts,patterns,shapes,chains,calc,backgrounds,fadings}


\setbeamersize{text margin left=15pt,text margin right=15pt,text margin bottom=15pt}


\begin{document}

\frame{\titlepage}

\setbeamerfont{frametitle}{size=\large}

\begin{frame}
\frametitle{Method}

\newcommand{\lw}{0.5mm}
% \newcommand{\mycirc}[2]{\draw (#1,#2) circle (3cm);}

\newcommand{\outFolder}{.}

\begin{columns}[T]
    \hspace{-3em}
    \begin{column}{.5\textwidth}
     %\begin{block}{}
    Model fitting with EM:
    \begin{enumerate}
    \item Estimate vertex assignment to clusters
    \item Refine trajectories
    \item Refine subject stages
    \end{enumerate}
    %\end{block}
    \end{column}
    \hspace{-5em}
    \begin{column}{.5\textwidth}
    %\begin{block}{}


    \begin{figure}
    \centering
    \begin{tikzpicture}
     \draw[line width=\lw] (-0.1,0) arc (-20:20:2) node (A1) {};
     \draw[line width=\lw] (1.7,0) arc (-20:20:2)  node (A2) {} ;
'''

deltaX = np.array([0, 0.1, 0.1, 0]) + 0.22
sigPos = [(1.2,3.0), (2.75, 2.25), (3, 0.8)]
colsAssignment = [[0,0,1,1],[0,2,0,1],[2,2,0,1],[0,0,1,2], [2,2,1,1], [1,2,2,0]]
latexCols = ['red', 'green', 'blue']

for i in range(4):
  for j in range(4):
    text += '     \draw[,fill=%s] (%.2f,%.2f) circle (0.15cm);\n' % \
            (latexCols[colsAssignment[j][i]], 0.4 *i + deltaX[j], 0.4 * j + 0.1 )


for g in range(3):
  text += '     \\node (sig%d) at (%.2f,%.2f) {\includegraphics[scale=0.08, trim=70 70 70 0]{\outFolder/sig%d.png}};\n' \
          % (g, sigPos[g][0], sigPos[g][1],g)

  text += '    \\node  (traj) at (sig%d.north) {\scriptsize{Traj. %d}}; \n' % (g,g)

text += r'''

    \draw[dotted, line width=0.6, color=red] (sig0.south west) -- (-0.1,1.5);
    \draw[dotted, line width=0.6, color=red] (sig0.south east) -- (1.8,1.5);

    \draw[dotted, line width=0.6, color=green] (sig1.west) -- (01,1.5);
    \draw[dotted, line width=0.6, color=green] (sig1.south) -- (1.8,0);

    \draw[dotted, line width=0.6, color=blue] (sig2.north west) -- (1.6,1.5);
    \draw[dotted, line width=0.6, color=blue] (sig2.south west) -- (0.7,0.3);

    \node  (vertices_label) at (0.8,1.8) {\scriptsize{Vertices}};


    % draw the bran and the magnification from it
    \node  (brain) at (-2,0.8) {\includegraphics[scale=0.06]{\outFolder/brain.png}};

    \draw (-1.6,1.1) circle (0.15cm) node (C) {};
    \draw[dotted, line width=0.6, color=black] (C.north) -- (A1.west);
    \draw[dotted, line width=0.6, color=black] (C.south) -- (-0.2,0);


   \end{tikzpicture}
  \end{figure}

    %\end{block}
    \end{column}
  \end{columns}

\end{frame}

\end{document}

'''

outFile = 'diagram.tex'
with open(outFile, 'w') as f:
  f.write(text)

os.system('pdflatex diagram.tex')
