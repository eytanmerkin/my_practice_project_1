"""
Advanced Quartile Predictor for Early-Season Win Predictions

This script demonstrates how early-season point differential can predict
final season outcomes better than win-loss record alone.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


class QuartilePredictor:
    """
    Predicts final season wins based on point differential at different
    stages of the season (quartiles).
    """
    
    def __init__(self, historical_data_path='nba_team_seasons_processed.csv'):
        """
        Initialize predictor with historical data.
        
        Args:
            historical_data_path: Path to processed NBA data CSV
        """
        self.df = pd.read_csv(historical_data_path)
        self.model = self._train_model()
    
    def _train_model(self):
        """Train the base linear regression model."""
        X = self.df[['POINT_DIFF_PER_GAME']].values
        y = self.df['WIN_PCT'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model
    
    def predict_final_wins(self, games_played, current_point_diff, total_games=82):
        """
        Predict final season wins based on current point differential.
        
        Args:
            games_played: Number of games played so far
            current_point_diff: Current point differential per game
            total_games: Total games in season (default 82)
        
        Returns:
            Dictionary with prediction details
        """
        # Calculate season progress
        season_pct = games_played / total_games
        
        # Predict win percentage based on current point diff
        predicted_win_pct = self.model.predict([[current_point_diff]])[0]
        predicted_win_pct = np.clip(predicted_win_pct, 0, 1)
        
        # Calculate expected final wins
        expected_wins = predicted_win_pct * total_games
        
        # Calculate confidence based on sample size
        # More games = higher confidence
        confidence = min(95, 50 + (season_pct * 50))
        
        # Calculate margin of error (decreases with more games)
        margin_of_error = (1 - season_pct) * 8  # Up to ±8 games early season
        
        return {
            'games_played': games_played,
            'season_progress_pct': season_pct * 100,
            'current_point_diff': current_point_diff,
            'predicted_win_pct': predicted_win_pct,
            'expected_wins': expected_wins,
            'expected_losses': total_games - expected_wins,
            'confidence_pct': confidence,
            'margin_of_error': margin_of_error,
            'win_range_low': max(0, expected_wins - margin_of_error),
            'win_range_high': min(total_games, expected_wins + margin_of_error)
        }
    
    def compare_point_diff_vs_record(self, games_played, wins, losses, point_diff):
        """
        Compare what point differential suggests vs actual record.
        
        Args:
            games_played: Number of games played
            wins: Current wins
            losses: Current losses
            point_diff: Current point differential per game
        
        Returns:
            Dictionary with comparison analysis
        """
        # Actual winning percentage
        actual_win_pct = wins / games_played
        
        # Expected winning percentage based on point diff
        expected_win_pct = self.model.predict([[point_diff]])[0]
        expected_win_pct = np.clip(expected_win_pct, 0, 1)
        
        # Calculate difference
        difference = actual_win_pct - expected_win_pct
        
        # Project to full season
        actual_pace_wins = actual_win_pct * 82
        expected_pace_wins = expected_win_pct * 82
        
        # Determine if over/underperforming
        if difference > 0.05:
            performance = "OVERPERFORMING"
            note = "Record is better than point differential suggests. May regress."
        elif difference < -0.05:
            performance = "UNDERPERFORMING"
            note = "Record is worse than point differential suggests. Likely to improve."
        else:
            performance = "ON PACE"
            note = "Record aligns with point differential."
        
        return {
            'actual_record': f"{wins}-{losses}",
            'actual_win_pct': actual_win_pct,
            'expected_win_pct': expected_win_pct,
            'difference': difference,
            'actual_pace_82_games': actual_pace_wins,
            'expected_pace_82_games': expected_pace_wins,
            'performance_status': performance,
            'note': note
        }
    
    def simulate_season_scenarios(self, games_played, current_point_diff, 
                                  num_simulations=1000):
        """
        Monte Carlo simulation of possible season outcomes.
        
        Args:
            games_played: Games played so far
            current_point_diff: Current point differential per game
            num_simulations: Number of simulations to run
        
        Returns:
            Dictionary with simulation results
        """
        games_remaining = 82 - games_played
        expected_win_pct = self.model.predict([[current_point_diff]])[0]
        expected_win_pct = np.clip(expected_win_pct, 0, 1)
        
        # Calculate current wins
        current_wins = int(expected_win_pct * games_played)
        
        # Simulate remaining games
        simulated_final_wins = []
        
        for _ in range(num_simulations):
            # Add some randomness to remaining games
            remaining_wins = np.random.binomial(games_remaining, expected_win_pct)
            final_wins = current_wins + remaining_wins
            simulated_final_wins.append(final_wins)
        
        simulated_final_wins = np.array(simulated_final_wins)
        
        return {
            'mean_wins': np.mean(simulated_final_wins),
            'median_wins': np.median(simulated_final_wins),
            'std_wins': np.std(simulated_final_wins),
            'min_wins': np.min(simulated_final_wins),
            'max_wins': np.max(simulated_final_wins),
            'percentile_25': np.percentile(simulated_final_wins, 25),
            'percentile_75': np.percentile(simulated_final_wins, 75),
            'probability_playoff': np.sum(simulated_final_wins >= 45) / num_simulations * 100,
            'probability_50_wins': np.sum(simulated_final_wins >= 50) / num_simulations * 100
        }
    
    def plot_prediction_evolution(self, point_diff_trajectory, actual_wins_trajectory=None):
        """
        Plot how predictions evolve throughout the season.
        
        Args:
            point_diff_trajectory: List of (games_played, point_diff) tuples
            actual_wins_trajectory: Optional list of (games_played, actual_wins)
        """
        predictions = []
        
        for games, pd in point_diff_trajectory:
            pred = self.predict_final_wins(games, pd)
            predictions.append({
                'games': games,
                'predicted_wins': pred['expected_wins'],
                'low': pred['win_range_low'],
                'high': pred['win_range_high']
            })
        
        pred_df = pd.DataFrame(predictions)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot prediction with confidence band
        ax.plot(pred_df['games'], pred_df['predicted_wins'], 
                'b-', linewidth=2, label='Predicted Wins')
        ax.fill_between(pred_df['games'], pred_df['low'], pred_df['high'],
                        alpha=0.3, label='Confidence Range')
        
        # Plot actual wins if provided
        if actual_wins_trajectory:
            actual_df = pd.DataFrame(actual_wins_trajectory, 
                                    columns=['games', 'actual_wins'])
            ax.plot(actual_df['games'], actual_df['actual_wins'],
                   'ro-', linewidth=2, label='Actual Wins', markersize=6)
        
        ax.set_xlabel('Games Played', fontsize=12, fontweight='bold')
        ax.set_ylabel('Expected Final Season Wins', fontsize=12, fontweight='bold')
        ax.set_title('Season Win Prediction Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 82)
        ax.set_ylim(0, 82)
        
        plt.tight_layout()
        plt.savefig('prediction_evolution.png', dpi=300, bbox_inches='tight')
        print("📊 Plot saved as 'prediction_evolution.png'")
        plt.show()


def demo_early_season_prediction():
    """
    Demonstrate the quartile predictor with example scenarios.
    """
    print("\n" + "="*70)
    print("NBA QUARTILE PREDICTOR - DEMO")
    print("="*70 + "\n")
    
    # Initialize predictor
    print("Loading historical data...")
    try:
        predictor = QuartilePredictor()
    except FileNotFoundError:
        print("⚠️  Run 'python pd_wp_analysis.py' first to generate historical data.")
        return
    
    print("✓ Model trained on historical data\n")
    
    # Scenario 1: Elite team after 20 games
    print("SCENARIO 1: Elite Team - 20 Games In")
    print("-" * 70)
    result = predictor.predict_final_wins(games_played=20, current_point_diff=8.5)
    print(f"Games Played: {result['games_played']} ({result['season_progress_pct']:.1f}% of season)")
    print(f"Point Differential: +{result['current_point_diff']:.1f} per game")
    print(f"Predicted Final Record: {result['expected_wins']:.1f}-{result['expected_losses']:.1f}")
    print(f"Win Range: {result['win_range_low']:.0f}-{result['win_range_high']:.0f} wins")
    print(f"Confidence: {result['confidence_pct']:.0f}%\n")
    
    # Scenario 2: Compare record vs point diff
    print("SCENARIO 2: Lucky Team? (Good Record, Mediocre Point Diff)")
    print("-" * 70)
    comparison = predictor.compare_point_diff_vs_record(
        games_played=25, wins=18, losses=7, point_diff=2.5
    )
    print(f"Actual Record: {comparison['actual_record']} ({comparison['actual_win_pct']:.1%})")
    print(f"Expected Win% (based on point diff): {comparison['expected_win_pct']:.1%}")
    print(f"Status: {comparison['performance_status']}")
    print(f"Actual pace: {comparison['actual_pace_82_games']:.1f} wins")
    print(f"Expected pace: {comparison['expected_pace_82_games']:.1f} wins")
    print(f"Note: {comparison['note']}\n")
    
    # Scenario 3: Monte Carlo simulation
    print("SCENARIO 3: Monte Carlo Simulation (1000 iterations)")
    print("-" * 70)
    sim_results = predictor.simulate_season_scenarios(
        games_played=30, current_point_diff=5.0, num_simulations=1000
    )
    print(f"Mean Final Wins: {sim_results['mean_wins']:.1f}")
    print(f"Median Final Wins: {sim_results['median_wins']:.0f}")
    print(f"Range: {sim_results['min_wins']:.0f} - {sim_results['max_wins']:.0f} wins")
    print(f"25th-75th Percentile: {sim_results['percentile_25']:.0f} - {sim_results['percentile_75']:.0f}")
    print(f"Playoff Probability (≥45 wins): {sim_results['probability_playoff']:.1f}%")
    print(f"50+ Win Probability: {sim_results['probability_50_wins']:.1f}%\n")
    
    # Scenario 4: Prediction evolution
    print("SCENARIO 4: Tracking Prediction Throughout Season")
    print("-" * 70)
    
    # Simulate a team's season trajectory
    trajectory = [
        (10, 4.2),   # After 10 games
        (20, 5.1),   # After 20 games
        (30, 5.8),   # Getting better
        (41, 6.2),   # Halfway point
        (60, 5.5),   # Slight decline
        (70, 5.9),   # Recovery
        (82, 6.0)    # Final
    ]
    
    print("Plotting prediction evolution...")
    predictor.plot_prediction_evolution(trajectory)
    
    print("\n" + "="*70)
    print("Demo complete! Check the generated plots.")
    print("="*70 + "\n")


def interactive_quartile_tool():
    """
    Interactive command-line tool for quartile predictions.
    """
    print("\n" + "="*70)
    print("INTERACTIVE QUARTILE PREDICTOR")
    print("="*70)
    
    try:
        predictor = QuartilePredictor()
    except FileNotFoundError:
        print("⚠️  Run 'python pd_wp_analysis.py' first to generate historical data.")
        return
    
    print("\nPredict final season outcomes based on early-season point differential.")
    print("Type 'demo' to run example scenarios, or 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("\nCommand [predict/compare/simulate/demo/quit]: ").strip().lower()
            
            if user_input in ['quit', 'q', 'exit']:
                print("Goodbye!")
                break
            
            elif user_input == 'demo':
                demo_early_season_prediction()
            
            elif user_input == 'predict':
                games = int(input("  Games played so far: "))
                point_diff = float(input("  Current point differential per game: "))
                
                result = predictor.predict_final_wins(games, point_diff)
                
                print(f"\n  📊 PREDICTION RESULTS")
                print(f"  Season Progress: {result['season_progress_pct']:.1f}%")
                print(f"  Predicted Final Wins: {result['expected_wins']:.1f}")
                print(f"  Expected Record: {result['expected_wins']:.0f}-{result['expected_losses']:.0f}")
                print(f"  Win Range: {result['win_range_low']:.0f}-{result['win_range_high']:.0f}")
                print(f"  Confidence: {result['confidence_pct']:.0f}%\n")
            
            elif user_input == 'compare':
                games = int(input("  Games played: "))
                wins = int(input("  Current wins: "))
                losses = int(input("  Current losses: "))
                point_diff = float(input("  Point differential per game: "))
                
                result = predictor.compare_point_diff_vs_record(games, wins, losses, point_diff)
                
                print(f"\n  📊 COMPARISON RESULTS")
                print(f"  Actual Record: {result['actual_record']} ({result['actual_win_pct']:.1%})")
                print(f"  Expected Win%: {result['expected_win_pct']:.1%}")
                print(f"  Status: {result['performance_status']}")
                print(f"  Projected Final Wins (current pace): {result['actual_pace_82_games']:.1f}")
                print(f"  Projected Final Wins (point diff): {result['expected_pace_82_games']:.1f}")
                print(f"  📝 {result['note']}\n")
            
            elif user_input == 'simulate':
                games = int(input("  Games played: "))
                point_diff = float(input("  Point differential per game: "))
                
                print("  Running 1000 simulations...")
                result = predictor.simulate_season_scenarios(games, point_diff)
                
                print(f"\n  📊 SIMULATION RESULTS")
                print(f"  Mean Final Wins: {result['mean_wins']:.1f}")
                print(f"  Median: {result['median_wins']:.0f}")
                print(f"  Range: {result['min_wins']:.0f}-{result['max_wins']:.0f}")
                print(f"  Playoff Probability: {result['probability_playoff']:.1f}%")
                print(f"  50+ Win Probability: {result['probability_50_wins']:.1f}%\n")
            
            else:
                print("  ⚠️  Unknown command. Use: predict, compare, simulate, demo, or quit")
        
        except ValueError as e:
            print(f"  ⚠️  Invalid input: {e}")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    # Run interactive tool
    interactive_quartile_tool()

