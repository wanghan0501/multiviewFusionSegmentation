"""
Created by Wang Han on 2021/5/25 20:40.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2021 Wang Han. SCU. All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy as np


def bland_altman_plot(m1,
                      m2,
                      sd_limit=1.96,
                      ax=None,
                      x_lims=[0, 1],
                      y_lims=[-1, 1],
                      fontsize=16,
                      scatter_kwds=None,
                      mean_line_kwds=None,
                      limit_lines_kwds=None):
    """
    Bland-Altman Plot.

    A Bland-Altman plot is a graphical method to analyze the differences
    between two methods of measurement. The mean of the measures is plotted
    against their difference.

    Parameters
    ----------
    m1, m2: pandas Series or array-like

    sd_limit : float, default 1.96
        The limit of agreements expressed in terms of the standard deviation of
        the differences. If `md` is the mean of the differences, and `sd` is
        the standard deviation of those differences, then the limits of
        agreement that will be plotted will be
                       md - sd_limit * sd, md + sd_limit * sd
        The default of 1.96 will produce 95% confidence intervals for the means
        of the differences.
        If sd_limit = 0, no limits will be plotted, and the ylimit of the plot
        defaults to 3 standard deviatons on either side of the mean.

    ax: matplotlib.axis, optional
        matplotlib axis object to plot on.

    scatter_kwargs: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.scatter plotting method

    mean_line_kwds: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method

    limit_lines_kwds: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method

   Returns
    -------
    ax: matplotlib Axis object
    """

    #     import numpy as np
    #     import matplotlib.pyplot as plt

    if len(m1) != len(m2):
        raise ValueError('m1 does not have the same length as m2.')
    if sd_limit < 0:
        raise ValueError('sd_limit ({}) is less than 0.'.format(sd_limit))

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)
    zeros = np.zeros_like(mean_diff)

    if ax is None:
        ax = plt.gca()

    scatter_kwds = scatter_kwds or {}
    if 's' not in scatter_kwds:
        scatter_kwds['s'] = 20
    mean_line_kwds = mean_line_kwds or {}
    limit_lines_kwds = limit_lines_kwds or {}
    for kwds in [mean_line_kwds, limit_lines_kwds]:
        if 'color' not in kwds:
            kwds['color'] = 'gray'
        if 'linewidth' not in kwds:
            kwds['linewidth'] = 1
    if 'linestyle' not in mean_line_kwds:
        kwds['linestyle'] = '--'
    if 'linestyle' not in limit_lines_kwds:
        kwds['linestyle'] = ':'

    ax.scatter(means, diffs, **scatter_kwds)
    ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.
    # ax.axhline(zeros, linestyle='--')  # draw mean line.
    plt.text(x_lims[0] + (x_lims[1] - x_lims[0]) * 0.9,
             mean_diff,
             'MEAN:\n{}'.format(np.round(mean_diff, 4)),
             va='center',
             ha='center',
             font={'size': fontsize})
    # Annotate mean line with mean difference.
    #     ax.annotate('MEAN:\n{}'.format(np.round(mean_diff, 4)),
    #                 xy=(0.99, 0.5),
    #                 horizontalalignment='right',
    #                 verticalalignment='center',
    #                 fontsize=14,
    #                 xycoords='axes fraction')

    if sd_limit > 0:
        half_ylim = (1.5 * sd_limit) * std_diff

        limit_of_agreement = sd_limit * std_diff

        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, **limit_lines_kwds)

        plt.text(x_lims[0] + (x_lims[1] - x_lims[0]) * 0.9,
                 lower,
                 '-SD{}:\n {}'.format(sd_limit, np.round(lower, 2)),
                 va='center',
                 ha='center',
                 font={'size': fontsize})
        plt.text(x_lims[0] + (x_lims[1] - x_lims[0]) * 0.9,
                 upper,
                 '+SD{}:\n {}'.format(sd_limit, np.round(upper, 2)),
                 va='center',
                 ha='center',
                 font={'size': fontsize})


    #         ax.annotate('-SD{}:\n {}'.format(sd_limit, np.round(lower, 4)),
    #                     xy=(0.99, lower),
    #                     horizontalalignment='right',
    #                     verticalalignment='bottom',
    #                     fontsize=14,
    #                     xycoords='axes points'
    #                    )
    #         ax.annotate('+SD{}:\n {}'.format(sd_limit, np.round(upper, 4)),
    #                     xy=(0.99, upper),
    #                     horizontalalignment='right',
    #                     fontsize=14,
    #                     xycoords='axes fraction'
    #                    )

    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        ax.set_ylim(mean_diff - half_ylim, mean_diff + half_ylim)

    # ax.set_ylabel('Difference', fontsize=14)
    # ax.set_xlabel('Average', fontsize=14)
    # ax.set_title("Bias: {}, 95% Limits of Agreement: {} to {}".format(np.round(mean_diff, 4), np.round(lower, 4), np.round(upper, 4)))
    plt.xlim(x_lims[0], x_lims[1])
    plt.xticks(np.linspace(x_lims[0], x_lims[1], 11))
    plt.ylim(y_lims[0], y_lims[1])
    plt.yticks(np.linspace(y_lims[0], y_lims[1], 11))
    plt.tight_layout()
    plt.tick_params(labelsize=fontsize)
    return ax
