#!/bin/bash

# Definition der Arrays (Indizes 0 bis 2)
pressure=(1 1 1 1 1 1 8 1)
deferral=(1 1 1 1 1 1 8 1)
utilization=(1 1 1 1 1 1 1 8)
underutil=(0 1 0 1 8 4 2 4)
shortage=(0 0 1 1 4 8 2 4)
bottleneck=(0 0 1 1 4 8 2 4)    # Nur 4 Werte übernommen

cd ..

# Schleife für die 8 Durchläufe
for i in {0..7}; do
    echo "========================================"
    echo "Starte Durchlauf $((i+1)) von 8"
    echo "Parameter:"
    echo "  - Pressure:          ${pressure[$i]}"
    echo "  - Deferral Priority: ${deferral[$i]}"
    echo "  - Bottleneck:        ${bottleneck[$i]}"
    echo "  - Utilization:       ${utilization[$i]}"
    echo "  - Underutilization:  ${underutil[$i]}"
    echo "  - Shortage:         ${shortage[$i]}"
    echo "========================================"

    # 1. Config.yaml anpassen
    source /.mt/bin/activate
    python scripts/update_config.py "${pressure[$i]}" "${deferral[$i]}" "${bottleneck[$i]}" "${utilization[$i]}" "${underutil[$i]}" "${shortage[$i]}"
    
    # Prüfen, ob das Update erfolgreich war
    if [ $? -ne 0 ]; then
        echo "Fehler beim Update der config.yaml. Breche ab."
        exit 1
    fi

    # 2. Eigentliche Simulation starten
    python scripts/run_simulation.py

    echo "Durchlauf $((i+1)) abgeschlossen."
    echo ""
done

echo "Alle Simulationen wurden erfolgreich beendet!"