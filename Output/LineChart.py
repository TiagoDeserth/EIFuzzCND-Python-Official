import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import os

from DebugLogger import DebugLogger

def plot_line_chart(application_title,
                    chart_title,
                    metricas,
                    rotulos_classificadores,
                    novidades,
                    novas_classes,
                    nome_dataset,
                    percented_label
                    ):
    #fig, ax = plt.subplots(figsize=(8,5))

    #DebugLogger.log(f"[DEBUG plot] novas_classes recebidas: {novas_classes}")
    #DebugLogger.log(f"[DEBUG plot] Total de linhas tracejadas a desenhar: {len(novas_classes)}")

    fig = plt.figure(figsize=(6.8, 4.5))
    gs = gridspec.GridSpec(2, 1, height_ratios = [1, 9])

    ax_top = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1], sharex = ax_top)

    cores = [
        (95/255, 173/255, 86/255),
        (242/255, 193/255, 78/255),
        (247/255, 129/255, 84/255),
        (49/255, 116/255, 161/255),
        (180/255, 67/255, 108/255)
    ]

    for i, valores in enumerate(metricas):
        cor = cores[i % len(cores)]
        ax_main.plot(range(1, len(valores) + 1), valores,
                label = rotulos_classificadores[i],
                linewidth = 2.5, color = cor)

    for classe in novas_classes:
        #DebugLogger.log(f"[DEBUG plot] Desenhando linha tracejada no momento {classe}")
        for ax in [ax_main, ax_top]:
            ax.axvline(x = classe, color = 'black', linestyle = '--', linewidth = 1.2)

    for i, novidade in enumerate(novidades):
        if novidade != 0.0:
            ax_main.axvline(x = i + 1, color = 'gray', linestyle = '-', linewidth = 1.2)

    ax_main.set_title(chart_title, fontsize = 14)
    ax_main.set_xlabel("Evaluation moments", fontsize = 12)
    ax_main.set_ylabel("Accuracy", fontsize = 12)
    ax_main.set_ylim(0, 105)
    ax_main.grid(False)
    ax_main.legend()

    if nome_dataset.lower() in ['moa', 'rbf']:
        x_interval = 5
    elif nome_dataset.lower() in ['synedc', 'kdd']:
        x_interval = 25
    elif nome_dataset.lower() == 'cover':
        x_interval = 50
    else:
        x_interval = 5

    #ax_main.xaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax_main.yaxis.set_major_locator(ticker.MultipleLocator(10))

    ax_main.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    ax_main.yaxis.set_major_locator(ticker.MultipleLocator(10))

    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([])
    ax_top.grid(False)
    ax_top.set_facecolor('white')

    ax_top.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
    ax_top.tick_params(axis = 'y', left = False)

    plt.tight_layout(h_pad = 0.2)

    caminho_graphics = os.path.join(".", "datasets", "graphics_data")
    os.makedirs(caminho_graphics, exist_ok = True)

    arquivo_saida = os.path.join(caminho_graphics, f"{nome_dataset}{application_title}-{percented_label}Python.png")
    plt.savefig(arquivo_saida, dpi = 100)
    #plt.close()

    return fig