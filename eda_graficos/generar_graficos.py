from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pulso_transmi import PulsoTransmiClient


OUTPUT = Path(__file__).parent


def main() -> None:
    with PulsoTransmiClient(timeout=60) as client:
        observations = client.observations_dataframe(page_size=5000)
        stations = client.stations()
        context = client.context_dataframe(page_size=5000)

    observations = observations.merge(stations[["station_id", "station_name"]], on="station_id")
    observations["local_at"] = observations["observed_at"].dt.tz_convert("America/Bogota")
    observations["local_hour"] = observations["local_at"].dt.hour
    observations["local_date"] = observations["local_at"].dt.date
    observations["weekday"] = observations["local_at"].dt.day_name()

    station_summary = (
        observations.groupby("station_id", as_index=False, observed=True)["demand"]
        .mean()
        .rename(columns={"demand": "mean_demand"})
        .merge(stations, on="station_id")
    )

    plt.style.use("seaborn-v0_8-whitegrid")

    hourly = observations.groupby("local_hour", as_index=False)["demand"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly["local_hour"], hourly["demand"], marker="o", color="#0b7285")
    ax.set(title="Demanda media por hora local", xlabel="Hora del día (Bogotá)", ylabel="Demanda media")
    ax.set_xticks(range(24))
    fig.tight_layout()
    fig.savefig(OUTPUT / "01_demanda_por_hora.png", dpi=160)
    plt.close(fig)

    station_order = observations.groupby("station_name")["demand"].mean().sort_values().index.tolist()
    observations["station_name"] = pd.Categorical(observations["station_name"], categories=station_order, ordered=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    observations.boxplot(column="demand", by="station_name", ax=ax, grid=False, rot=45)
    ax.set(title="Distribución de demanda por estación", xlabel="Estación", ylabel="Demanda")
    fig.suptitle("")
    ax.set_xticklabels(station_order, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUTPUT / "02_distribucion_por_estacion.png", dpi=160)
    plt.close(fig)

    daily = observations.groupby(["local_date", "station_name"], as_index=False, observed=True)["demand"].sum()
    total_daily = daily.groupby("local_date", as_index=False)["demand"].sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(total_daily["local_date"], total_daily["demand"], color="#d9480f", linewidth=1.5)
    ax.set(title="Demanda total diaria", xlabel="Fecha local", ylabel="Demanda acumulada")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(OUTPUT / "03_demanda_diaria.png", dpi=160)
    plt.close(fig)

    pivot = observations.pivot_table(index="station_name", columns="local_hour", values="demand", aggfunc="mean", observed=True)
    fig, ax = plt.subplots(figsize=(13, 6))
    image = ax.imshow(pivot, aspect="auto", cmap="YlOrRd")
    ax.set(title="Mapa de calor de demanda media", xlabel="Hora local", ylabel="Estación")
    ax.set_xticks(range(24), range(24))
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    fig.colorbar(image, ax=ax, label="Demanda media")
    fig.tight_layout()
    fig.savefig(OUTPUT / "04_mapa_calor_estacion_hora.png", dpi=160)
    plt.close(fig)

    context_plot = context.set_index("observed_at").sort_index()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(context_plot.index, context_plot["rain_mm"], label="Lluvia observada", color="#1971c2")
    axes[0].plot(context_plot.index, context_plot["rain_forecast"], label="Pronóstico lluvia", color="#74c0fc", alpha=0.8)
    axes[0].set_ylabel("mm")
    axes[0].legend()
    axes[1].plot(context_plot.index, context_plot["temperature_c"], label="Temperatura observada", color="#e8590c")
    axes[1].plot(context_plot.index, context_plot["temperature_forecast"], label="Pronóstico temperatura", color="#ffa94d", alpha=0.8)
    axes[1].set_ylabel("°C")
    axes[1].legend()
    fig.suptitle("Contexto observado y pronosticado")
    fig.tight_layout()
    fig.savefig(OUTPUT / "05_contexto_observado_vs_pronostico.png", dpi=160)
    plt.close(fig)

    merged = observations[["observed_at", "demand"]].merge(context, on="observed_at")
    correlations = merged[["demand", "rain_mm", "temperature_c", "event_intensity"]].corr()["demand"].drop("demand").sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    correlations.plot.barh(ax=ax, color="#6741d9")
    ax.set(title="Correlación simple con demanda", xlabel="Correlación de Pearson", ylabel="Variable")
    fig.tight_layout()
    fig.savefig(OUTPUT / "06_correlacion_contexto_demanda.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    corridors = station_summary["corridor"].unique().tolist()
    colors = plt.get_cmap("tab10").colors
    for index, corridor in enumerate(corridors):
        group = station_summary[station_summary["corridor"] == corridor]
        ax.scatter(
            group["longitude"],
            group["latitude"],
            s=group["mean_demand"] * 1.5,
            color=colors[index % len(colors)],
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
            label=corridor,
        )
        for _, station in group.iterrows():
            ax.annotate(
                station["station_id"],
                (station["longitude"], station["latitude"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set(title="Geolocalización de estaciones", xlabel="Longitud", ylabel="Latitud")
    ax.legend(title="Corredor")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(OUTPUT / "07_mapa_geolocalizacion_estaciones.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()