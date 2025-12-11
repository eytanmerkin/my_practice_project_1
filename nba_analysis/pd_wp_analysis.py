"""
NBA Point Differential vs Winning Percentage Analysis
Fetches NBA team season data and analyzes the relationship between point differential and winning percentage.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from nba_api.stats.endpoints import leaguegamefinder, leaguedashteamstats
# Note: teams.get_teams() only returns current team IDs, which excludes historical teams
import time
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def fetch_team_season_data(seasons: List[str] = None, num_seasons: int = 30) -> pd.DataFrame:
    """
    Fetch team season data from NBA API.
    
    Args:
        seasons: List of season strings (e.g., ['2022-23', '2021-22']). 
                 If None, fetches last num_seasons seasons.
        num_seasons: Number of seasons to fetch if seasons is None (default: 30).
                     More seasons = more data = better model accuracy.
    
    Returns:
        DataFrame with team season statistics (NBA teams only)
    """
    if seasons is None:
        # Start from 1996-97 (earliest reliable data in NBA API for this endpoint)
        # to 2024-25 current season
        start_year = 1996
        end_year = 2025
        seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(start_year, end_year)]
    
    all_data = []
    
    print(f"Fetching data for {len(seasons)} seasons...")
    
    for season in seasons:
        try:
            print(f"  Fetching {season}...", end=" ")
            
            # Fetch team stats for the season
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star='Regular Season',
                per_mode_detailed='Totals'
            )
            
            df = team_stats.get_data_frames()[0]
            
            # Filter to NBA teams only using team ID range
            # NBA team IDs are in the 1610612700-1610612799 range
            # This excludes G-League (12XX), WNBA (161161XXXX), and other leagues
            df = df[df['TEAM_ID'].between(1610612700, 1610612799)].copy()
            
            # Filter out teams with too few games (likely data errors or partial seasons)
            # Require at least 20 games (to handle lockout seasons like 2011-12, 1998-99)
            df = df[df['GP'] >= 20].copy()
            
            if len(df) > 0:
                df['SEASON'] = season
                all_data.append(df)
                print(f"✓ ({len(df)} NBA teams)")
            else:
                print(f"⚠ (no data or incomplete season)")
            
            # Rate limiting to avoid API throttling
            time.sleep(0.6)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data was fetched. Please check your connection and try again.")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal NBA team-seasons fetched: {len(combined_df)}")
    
    return combined_df


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate point differential, winning percentage, and expected wins.
    
    Args:
        df: DataFrame with team season statistics
    
    Returns:
        DataFrame with calculated metrics
    """
    # Validate required columns
    required_cols = ['PTS', 'PLUS_MINUS', 'GP', 'W', 'L']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found: {missing_cols}")
    
    # Use GP (games played) from API - it's more reliable
    df['GAMES_PLAYED'] = df['GP'].astype(float)
    
    # Calculate total points scored and allowed
    # PTS is total points scored
    # PLUS_MINUS = PTS - OPP_PTS, so OPP_PTS = PTS - PLUS_MINUS
    df['TOTAL_PTS_SCORED'] = df['PTS'].astype(float)
    df['TOTAL_PTS_ALLOWED'] = (df['PTS'] - df['PLUS_MINUS']).astype(float)
    df['POINT_DIFFERENTIAL'] = df['PLUS_MINUS'].astype(float)
    
    # Calculate winning percentage (use W_PCT from API if available, otherwise calculate)
    if 'W_PCT' in df.columns:
        df['WIN_PCT'] = df['W_PCT'].astype(float)
    else:
        # Calculate from wins and losses
        df['WIN_PCT'] = df['W'].astype(float) / (df['W'].astype(float) + df['L'].astype(float))
    
    # Filter out any invalid data (NaN, inf, or unrealistic values)
    df = df[df['GAMES_PLAYED'] > 0].copy()
    df = df[df['WIN_PCT'].between(0, 1)].copy()
    
    # Calculate per-game metrics
    df['PPG'] = df['TOTAL_PTS_SCORED'] / df['GAMES_PLAYED']
    df['OPP_PPG'] = df['TOTAL_PTS_ALLOWED'] / df['GAMES_PLAYED']
    df['POINT_DIFF_PER_GAME'] = df['POINT_DIFFERENTIAL'] / df['GAMES_PLAYED']
    
    # Filter out unrealistic point differentials (more than ±20 per game is extremely rare)
    df = df[df['POINT_DIFF_PER_GAME'].between(-20, 20)].copy()
    
    return df


def perform_linear_regression(df: pd.DataFrame) -> Tuple[LinearRegression, Dict]:
    """
    Perform linear regression between point differential per game and winning percentage.
    
    Args:
        df: DataFrame with calculated metrics
    
    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    # Prepare data
    X = df[['POINT_DIFF_PER_GAME']].values
    y = df['WIN_PCT'].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'slope': model.coef_[0],
        'intercept': model.intercept_
    }
    
    print("\n" + "="*60)
    print("LINEAR REGRESSION RESULTS")
    print("="*60)
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Equation: Win% = {model.intercept_:.4f} + {model.coef_[0]:.4f} × Point Diff")
    print(f"\nInterpretation: Each point of differential = {model.coef_[0]*82:.2f} wins over 82 games")
    print("="*60 + "\n")
    
    return model, metrics


def plot_analysis(df: pd.DataFrame, model: LinearRegression, metrics: Dict):
    """
    Create visualization of point differential vs winning percentage.
    
    Args:
        df: DataFrame with metrics
        model: Trained linear regression model
        metrics: Dictionary of model metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Scatter plot with regression line
    X_plot = np.linspace(df['POINT_DIFF_PER_GAME'].min(), 
                         df['POINT_DIFF_PER_GAME'].max(), 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    
    scatter = ax1.scatter(df['POINT_DIFF_PER_GAME'], df['WIN_PCT'], 
                         alpha=0.6, s=50, c=df['SEASON'].astype('category').cat.codes,
                         cmap='viridis')
    ax1.plot(X_plot, y_plot, 'r--', linewidth=2, label='Linear Regression')
    
    ax1.set_xlabel('Point Differential per Game', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Winning Percentage', fontsize=12, fontweight='bold')
    ax1.set_title('Point Differential vs Winning Percentage\n(All NBA Team Seasons)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add R² annotation
    ax1.text(0.05, 0.95, f'R² = {metrics["r2"]:.4f}\nRMSE = {metrics["rmse"]:.4f}',
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Distribution of point differentials
    ax2.hist(df['POINT_DIFF_PER_GAME'], bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Differential')
    ax2.set_xlabel('Point Differential per Game', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Point Differentials', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nba_point_diff_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Plot saved as 'nba_point_diff_analysis.png'")
    plt.show()


def predict_wins(model: LinearRegression, point_diff: float, games: int = 82) -> Dict:
    """
    Predict expected wins based on point differential.
    
    Args:
        model: Trained linear regression model
        point_diff: Point differential per game
        games: Number of games in season (default 82)
    
    Returns:
        Dictionary with prediction results
    """
    win_pct = model.predict([[point_diff]])[0]
    win_pct = np.clip(win_pct, 0, 1)  # Ensure between 0 and 1
    expected_wins = win_pct * games
    
    return {
        'point_diff': point_diff,
        'win_pct': win_pct,
        'expected_wins': expected_wins,
        'expected_losses': games - expected_wins
    }


def interactive_predictor(model: LinearRegression):
    """
    Interactive command-line tool for predicting wins based on point differential.
    
    Args:
        model: Trained linear regression model
    """
    print("\n" + "="*60)
    print("INTERACTIVE WIN PREDICTOR")
    print("="*60)
    print("Enter point differential per game to predict expected wins.")
    print("Type 'quit' or 'q' to exit.\n")
    
    while True:
        try:
            user_input = input("Point differential per game: ").strip()
            
            if user_input.lower() in ['quit', 'q', 'exit']:
                print("Exiting predictor. Goodbye!")
                break
            
            point_diff = float(user_input)
            
            # Predict for 82 game season
            result = predict_wins(model, point_diff, 82)
            
            print(f"\n  📈 Predicted Win%: {result['win_pct']:.1%}")
            print(f"  🏆 Expected Wins: {result['expected_wins']:.1f}")
            print(f"  📉 Expected Losses: {result['expected_losses']:.1f}")
            print(f"  📊 Record: {result['expected_wins']:.0f}-{result['expected_losses']:.0f}\n")
            
        except ValueError:
            print("  ⚠️  Please enter a valid number.\n")
        except KeyboardInterrupt:
            print("\n\nExiting predictor. Goodbye!")
            break


def calculate_quartile_analysis(df: pd.DataFrame, model: LinearRegression) -> pd.DataFrame:
    """
    Analyze point differential by quartiles throughout the season.
    This requires game-by-game data, which we'll simulate based on season totals.
    
    Args:
        df: DataFrame with team season data
        model: Trained regression model
    
    Returns:
        DataFrame with quartile analysis
    """
    print("\n" + "="*60)
    print("QUARTILE ANALYSIS")
    print("="*60)
    print("Note: This analysis uses season-long data.")
    print("For true quartile analysis, game-by-game data would be needed.")
    print("="*60 + "\n")
    
    # For demonstration, we'll show how early-season point diff predicts final wins
    # In a real scenario, you'd need game-by-game data
    
    quartile_results = []
    
    for _, row in df.iterrows():
        games_played = row['GAMES_PLAYED']
        
        # Simulate quartile breakdowns (in real scenario, use actual game data)
        q1_games = int(games_played * 0.25)
        q2_games = int(games_played * 0.50)
        q3_games = int(games_played * 0.75)
        
        quartile_results.append({
            'TEAM_NAME': row['TEAM_NAME'],
            'SEASON': row['SEASON'],
            'ACTUAL_WINS': row['W'],
            'POINT_DIFF_PER_GAME': row['POINT_DIFF_PER_GAME'],
            'Q1_GAMES': q1_games,
            'Q2_GAMES': q2_games,
            'Q3_GAMES': q3_games,
            'TOTAL_GAMES': games_played
        })
    
    quartile_df = pd.DataFrame(quartile_results)
    
    print(f"Sample quartile breakdown (first 5 teams):")
    print(quartile_df.head())
    
    return quartile_df


def main():
    """
    Main execution function.
    """
    print("\n" + "="*60)
    print("NBA POINT DIFFERENTIAL vs WINNING PERCENTAGE ANALYSIS")
    print("="*60 + "\n")
    
    # Fetch data - using all available seasons (1996-2025) for maximum statistical power
    # This gives us ~850+ team-seasons for robust analysis
    print("Step 1: Fetching NBA season data...")
    print("Note: Fetching seasons from 1996-2025 (~850+ team-seasons)")
    print("      NBA API doesn't have reliable data before 1996-97 for this endpoint")
    print("      This may take several minutes due to API rate limiting...")
    df = fetch_team_season_data()
    
    # Calculate metrics
    print("\nStep 2: Calculating metrics...")
    df = calculate_metrics(df)
    
    # Perform regression
    print("\nStep 3: Performing linear regression...")
    model, metrics = perform_linear_regression(df)
    
    # Create visualizations
    print("\nStep 4: Creating visualizations...")
    plot_analysis(df, model, metrics)
    
    # Quartile analysis
    print("\nStep 5: Quartile analysis...")
    quartile_df = calculate_quartile_analysis(df, model)
    
    # Save processed data
    df.to_csv('nba_team_seasons_processed.csv', index=False)
    print("\n💾 Processed data saved as 'nba_team_seasons_processed.csv'")
    
    # Interactive predictor
    print("\nStep 6: Interactive predictor...")
    interactive_predictor(model)


if __name__ == "__main__":
    main()
