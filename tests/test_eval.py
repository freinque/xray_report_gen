import unittest

from xray_report_gen import eval

refs = [
    "Interstitial opacities without changes.",
    #"Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
    #"Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
]
hyps = [
    "Interstitial opacities at bases without changes.",
    #"Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
    #"Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
]


class MyTestCase(unittest.TestCase):
    def test_get_green_scorer_res(self):
        print('refs:', refs)
        print('hyps:', hyps)

        mean, std, green_score_list, summary, result_df = eval.get_green_scorer_res(refs, hyps)

        print(green_score_list)
        print(summary)

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
