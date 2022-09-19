import numpy as np
import matplotlib.pyplot as plt

resultats = {
    "MacBook Pro": [108.2, 2.43, 7.69, 25.05, 39.7, 33.33],
    "iPad Pro": [142.12, 3.02, 8.88, 29.62, 51.81, 48.79],
    "iPhone XR": [222.53, 2.92, 10.02, 35.57, 78.95, 95.06],
    "Raspberry Pi 4": [559.23, 17.74, 43.18, 120.95, 192.02, 185.34],
}

# set width of bar
barWidth = 0.20
fig = plt.subplots(figsize =(8, 5), num="Comparaison des performances de différentes machines")

# set height of bar
MB = resultats["MacBook Pro"]
IPA = resultats["iPad Pro"]
IPO = resultats["iPhone XR"]
RPI = resultats["Raspberry Pi 4"]
 
# Set position of bar on X axis
br1 = np.arange(len(MB))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

plt.bar(br1, MB, color = 'b', width = barWidth, label ='MacBook Pro (Intel i7)')
plt.bar(br2, IPA, color ='g', width = barWidth, label ='iPad Pro (Apple A10X)')
plt.bar(br3, IPO, color ='y', width = barWidth, label ='iPhone XR (Apple A12)')
plt.bar(br4, RPI, color ='r', width = barWidth, label ='Raspberry Pi 4 (ARM Cortex A72)')
 
plt.xlabel("Le plus bas est le mieux", fontweight="bold")
plt.ylabel("Temps d'exécution (secondes)", fontweight="bold")
plt.xticks([r + 1.5*barWidth for r in range(len(MB))], ['Temps total', 'Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5'])

plt.title("Comparaison des performances de différentes machines", fontweight="bold")
plt.legend()
plt.show()