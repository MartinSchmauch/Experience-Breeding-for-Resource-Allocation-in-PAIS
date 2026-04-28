import yaml
import sys
import os
from pathlib import Path

def main():
    # Wir erwarten 7 Argumente: Skriptname + 6 Parameter
    if len(sys.argv) != 7:
        print("Fehler: Erwarte genau 6 Parameter für die Config.")
        sys.exit(1)

    _, pressure, deferral, bottleneck, utilization, underutil, shortage = sys.argv

    _PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    print(_PROJECT_ROOT)
    config_file = Path(_PROJECT_ROOT) / "config" / "simulation_config.yaml"
    if config_file.exists():
        print(f"Config-Datei gefunden: {config_file}")
    # YAML Datei sicher laden
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Fehler: {config_file} nicht gefunden.")
        sys.exit(1)

    # Werte überschreiben (als float gecastet)
    weights = config['optimization']['objective_weights']
    weights['pressure'] = float(pressure)
    weights['deferral_priority'] = float(deferral)
    weights['utilization'] = float(utilization)
    weights['bottleneck'] = float(bottleneck)
    weights['underutilization'] = float(underutil)
    weights['shortage'] = float(shortage)
    if bottleneck == '0':
        config['mentoring']['bottleneck_activity_strategy']['enabled'] = 0  # Deaktivieren, wenn Gewicht 0 ist
    else:
        config['mentoring']['bottleneck_activity_strategy']['enabled'] = 1  # Aktivieren, wenn Gewicht > 0 ist
    if shortage == '0':
        config['mentoring']['same_day_shortage_strategy']['enabled'] = 0  # Deaktivieren, wenn Gewicht 0 ist
    else:
        config['mentoring']['same_day_shortage_strategy']['enabled'] = 1  # Aktivieren, wenn Gewicht > 0 ist
    if underutil == '0':
        config['mentoring']['underutilization_strategy']['enabled'] = 0  # Deaktivieren, wenn Gewicht 0 ist
    else:
        config['mentoring']['underutilization_strategy']['enabled'] = 1  # Aktivieren, wenn Gewicht > 0 ist

    # YAML wieder speichern
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    main()