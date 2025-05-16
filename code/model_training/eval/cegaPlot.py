import matplotlib.pyplot as plt


class ClarkeErrorGridAnalyzer:
    """
    A class for performing Clarke Error Grid (CEG) analysis to evaluate the clinical accuracy 
    of glucose predictions compared to reference values.
    """

    def __init__(self, model_name, log_file):
        """
        Initializes the ClarkeErrorGridAnalyzer with color mapping for zones, counters for zone assignments,
        and data containers for plotting.
        """
        self.model_name = model_name
        self.log_file = log_file
        self.zone_colors = {
            0: '#6BCB77',  # Zone A
            1: '#F0C04A',  # Zone B
            2: '#FFAC76',  # Zone C
            3: '#A77BCA',  # Zone D
            4: '#FF6F61'   # Zone E
        }
        self.zone_counts = [0] * 5
        self.plot_data = {i: {'x': [], 'y': []} for i in range(5)}
        self.ref_values = None
        self.pred_values = None

    def classify_points(self, ref_values, pred_values):
        """
        Classifies each prediction-reference pair into Clarke Error Grid zones (A to E)
        and logs the results to a summary file.

        Parameters:
        - ref_values (list or array-like): List of reference glucose values (mg/dL).
        - pred_values (list or array-like): Corresponding list of predicted glucose values (mg/dL).
        - model_name (str): Optional model name to include in the summary file.
        - log_file (str): Path to the text file for logging CEG summary results.
        """

        self.ref_values = ref_values
        self.pred_values = pred_values

        # Reset previous state
        self.zone_counts = [0] * 5
        self.plot_data = {i: {'x': [], 'y': []} for i in range(5)}

        for ref, pred in zip(ref_values, pred_values):
            if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
                self._add_point(0, ref, pred)  # Zone A
            elif (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
                self._add_point(4, ref, pred)  # Zone E
            elif (70 <= ref <= 290 and pred >= ref + 110) or (130 <= ref <= 180 and pred <= (7 / 5) * ref - 182):
                self._add_point(2, ref, pred)  # Zone C
            elif (ref >= 240 and 70 <= pred <= 180) or (ref <= 175 / 3 and 70 <= pred <= 180) or (175 / 3 <= ref <= 70 and pred >= (6 / 5) * ref):
                self._add_point(3, ref, pred)  # Zone D
            else:
                self._add_point(1, ref, pred)  # Zone B

        self.print_summary()

    def _add_point(self, zone_id, ref, pred):
        """
        Adds a data point to the appropriate zone count and stores it for plotting.

        Parameters:
        - zone_id (int): Zone index (0=A, 1=B, 2=C, 3=D, 4=E).
        - ref (float): Reference glucose value (mg/dL).
        - pred (float): Predicted glucose value (mg/dL).
        """
        self.zone_counts[zone_id] += 1
        self.plot_data[zone_id]['x'].append(ref)
        self.plot_data[zone_id]['y'].append(pred)

    def print_summary(self):
        """
        Prints the percentage of points in each CEG zone and the combined A+B accuracy.
        Optionally logs the same summary to a file.

        Parameters:
        - model_name (str): Name of the model for labeling the output summary.
        - log_file (str or None): If provided, path to a text file to log the summary.
        """
        total_points = len(self.ref_values)
        output_lines = [f"Model: {self.model_name}"]

        for i, count in enumerate(self.zone_counts):
            percentage = (count / total_points) * 100
            output_lines.append(f"Zone {chr(65 + i)}: {percentage:.2f}%")

        total_ab = self.zone_counts[0] + self.zone_counts[1]
        percentage_ab = (total_ab / total_points) * 100
        output_lines.append(f"Zones (A + B): {percentage_ab:.2f}%\n")

        # Print to console
        for line in output_lines:
            print(line)

        # Log to file
        with open(self.log_file, "a", encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + "\n")

    def plot(self, title_string="", zone_fontsize=14, label_fontsize=12, title_fontsize=15):
        """
        Creates and displays the Clarke Error Grid plot with zone boundaries and labels.

        Parameters:
        - model_name (str): Name of the model to be included in the plot title.
        - title_string (str): Optional custom string to prefix the plot title.
        - zone_fontsize (int): Font size for zone labels.
        - label_fontsize (int): Font size for axis labels.
        - title_fontsize (int): Font size for the plot title.
        """
        for i in range(4, -1, -1):
            plt.scatter(self.plot_data[i]['x'], self.plot_data[i]['y'], marker='.', s=50,
                        c=self.zone_colors[i], label=f'Zone {chr(65 + i)}')

        plt.title(title_string +
                  f" CEG - {self.model_name}", fontsize=title_fontsize)
        plt.xlabel("Reference  concentration (mg/dL)", fontsize=label_fontsize)
        plt.ylabel("Predicted concentration (mg/dL)", fontsize=label_fontsize)
        plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
        plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
        plt.gca().set_facecolor('white')
        plt.gca().set_xlim([0, 400])
        plt.gca().set_ylim([0, 400])
        plt.gca().set_aspect(1)

        # Zone lines
        plt.plot([0, 400], [0, 400], ':', c='black')
        plt.plot([0, 175 / 3], [70, 70], '-', c='black')
        plt.plot([175 / 3, 400 / 1.2], [70, 400], '-', c='black')
        plt.plot([70, 70], [84, 400], '-', c='black')
        plt.plot([0, 70], [180, 180], '-', c='black')
        plt.plot([70, 290], [180, 400], '-', c='black')
        plt.plot([70, 70], [0, 56], '-', c='black')
        plt.plot([70, 400], [56, 320], '-', c='black')
        plt.plot([180, 180], [0, 70], '-', c='black')
        plt.plot([180, 400], [70, 70], '-', c='black')
        plt.plot([240, 240], [70, 180], '-', c='black')
        plt.plot([240, 400], [180, 180], '-', c='black')
        plt.plot([130, 180], [0, 70], '-', c='black')

        # Zone labels
        plt.text(30, 15, "A", fontsize=zone_fontsize)
        plt.text(370, 260, "B", fontsize=zone_fontsize)
        plt.text(280, 370, "B", fontsize=zone_fontsize)
        plt.text(160, 370, "C", fontsize=zone_fontsize)
        plt.text(160, 15, "C", fontsize=zone_fontsize)
        plt.text(30, 140, "D", fontsize=zone_fontsize)
        plt.text(370, 120, "D", fontsize=zone_fontsize)
        plt.text(30, 370, "E", fontsize=zone_fontsize)
        plt.text(370, 15, "E", fontsize=zone_fontsize)

    def save_plot(self, filename="ceg_plot.png"):
        """
        Saves the current CEG plot to a file.

        Parameters:
        - filename (str): Path or filename to save the plot image. Default is 'ceg_plot.png'.
        """
        plt.savefig(filename, dpi=300, bbox_inches='tight')
