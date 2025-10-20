"""
Interactive Household Trajectory Viewer

Shows all individuals within a household using compact detailed view.
Navigate between households sorted by total infection count.

Features:
- One household per view (all members shown)
- Households sorted by total infection count
- Previous/Next buttons to navigate between households
- Compact detailed view with annotations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from plotting_utils import plot_individual_compact

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Site selection (change this to view different sites)
SITE = 'nagongera'  # Options: 'nagongera', 'walukuba', 'kihihi'

# ==============================================================================

print(f"Loading PRISM data for {SITE.upper()} site...")
df = pd.read_csv(f'data/prism_cleaned_{SITE}.csv', parse_dates=['date'])
print(f"Loaded {len(df)} observations for {df['id'].nunique()} participants")

# Get household statistics
household_stats = df.groupby('Household_Id').agg({
    'id': 'nunique',  # Number of household members
    'parasitedensity': lambda x: (x > 0).sum()  # Total infection events
}).rename(columns={
    'id': 'n_members',
    'parasitedensity': 'total_infections'
})

# Filter to households with at least 2 members and some infections
multi_person = household_stats[
    (household_stats['n_members'] >= 2) &
    (household_stats['total_infections'] > 0)
].sort_values('total_infections', ascending=False)

print(f"\nFound {len(multi_person)} multi-person households with infections")
print(f"Household size range: {household_stats['n_members'].min():.0f}-{household_stats['n_members'].max():.0f}")
print(f"Total infections per household range: {multi_person['total_infections'].min():.0f}-{multi_person['total_infections'].max():.0f}")


class HouseholdViewer:
    def __init__(self, data, household_list):
        self.df = data
        self.households = household_list.index.tolist()
        self.household_stats = household_list
        self.current_idx = 0

        # Store global date range for consistent x-axis across all households
        # Add 3-month buffer on each end to prevent label truncation
        self.date_min = data['date'].min() - pd.DateOffset(months=3)
        self.date_max = data['date'].max() + pd.DateOffset(months=3)

        # Create figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(20, 10))
        self.fig.subplots_adjust(top=0.85)

        # Navigation buttons (moved to top)
        ax_prev_20 = plt.axes([0.20, 0.94, 0.08, 0.04])
        ax_prev = plt.axes([0.29, 0.94, 0.10, 0.04])
        ax_next = plt.axes([0.61, 0.94, 0.10, 0.04])
        ax_next_20 = plt.axes([0.72, 0.94, 0.08, 0.04])

        self.btn_prev_20 = Button(ax_prev_20, '<<<  -20')
        self.btn_prev = Button(ax_prev, '<  Previous')
        self.btn_next = Button(ax_next, 'Next  >')
        self.btn_next_20 = Button(ax_next_20, '+20  >>>')

        # Use direct mouse click handler for better responsiveness
        # (don't use button.on_clicked to avoid double-trigger)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        # Info text area (moved to top)
        self.ax_info = plt.axes([0.02, 0.94, 0.15, 0.04])
        self.ax_info.axis('off')

        # Keyboard navigation
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Plot first household
        self.plot_household()

    def plot_household(self):
        """Plot current household"""
        self.ax.clear()

        household_id = self.households[self.current_idx]
        household_data = self.df[self.df['Household_Id'] == household_id].copy()

        # Get household members
        members = household_data['id'].unique()
        n_members = len(members)
        total_infections = self.household_stats.loc[household_id, 'total_infections']

        # Get age and infection info for each member
        member_info = []
        for member_id in members:
            member_data = household_data[household_data['id'] == member_id]
            age = member_data['age_at_enrollment'].iloc[0]
            gender = member_data['gender'].iloc[0]
            n_infections = (member_data['parasitedensity'] > 0).sum()
            member_info.append({
                'id': member_id,
                'age': age,
                'gender': gender,
                'n_infections': n_infections
            })

        member_info_df = pd.DataFrame(member_info)

        # Plot using compact detailed view
        plot_individual_compact(household_data, ax=self.ax, fig=self.fig, detailed=True)

        # Set consistent x-axis range for all households
        self.ax.set_xlim(self.date_min, self.date_max)

        # Add household ID label in upper right corner
        self.ax.text(0.98, 0.98, f'Household {household_id}',
                    transform=self.ax.transAxes,
                    fontsize=14, fontweight='bold',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, edgecolor='black', linewidth=1.5))

        # Title with household info
        title = f'Household {household_id} ({self.current_idx + 1}/{len(self.households)})\n'
        title += f'{n_members} members, {int(total_infections)} total infection events\n'
        title += 'Members: ' + ', '.join([
            f"ID{row['id']}({int(row['age'])}{row['gender'][0]}, {int(row['n_infections'])}+)"
            for _, row in member_info_df.iterrows()
        ])
        self.ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

        # Update info text
        self.ax_info.clear()
        self.ax_info.axis('off')
        info_text = f"Household {self.current_idx + 1} of {len(self.households)}"
        self.ax_info.text(0.5, 0.5, info_text, ha='center', va='center',
                         fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        self.fig.canvas.draw()

    def next_household(self, event=None):
        """Go to next household"""
        if self.current_idx < len(self.households) - 1:
            self.current_idx += 1
            self.plot_household()

    def prev_household(self, event=None):
        """Go to previous household"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.plot_household()

    def next_20_households(self, event=None):
        """Skip forward 20 households"""
        self.current_idx = min(self.current_idx + 20, len(self.households) - 1)
        self.plot_household()

    def prev_20_households(self, event=None):
        """Skip backward 20 households"""
        self.current_idx = max(self.current_idx - 20, 0)
        self.plot_household()

    def on_mouse_click(self, event):
        """Handle direct mouse clicks for better button responsiveness"""
        # Check if click is within button bounds and trigger action
        # This provides better responsiveness than the default button widget
        if event.inaxes:
            # Get click coordinates
            x, y = event.x, event.y
            # Convert to figure coordinates
            fig_x = x / self.fig.get_dpi() / self.fig.get_figwidth()
            fig_y = y / self.fig.get_dpi() / self.fig.get_figheight()

            # Check which button was clicked (approximate bounds) - buttons now at top
            if 0.94 < fig_y < 0.98:  # Button row at top
                if 0.20 < fig_x < 0.28:
                    self.prev_20_households()
                elif 0.29 < fig_x < 0.39:
                    self.prev_household()
                elif 0.61 < fig_x < 0.71:
                    self.next_household()
                elif 0.72 < fig_x < 0.80:
                    self.next_20_households()

    def on_key(self, event):
        """Handle keyboard navigation"""
        if event.key == 'right':
            self.next_household()
        elif event.key == 'left':
            self.prev_household()
        elif event.key == 'q':
            plt.close(self.fig)


print("\n" + "=" * 80)
print("INTERACTIVE HOUSEHOLD VIEWER CONTROLS")
print("=" * 80)
print("\nNavigation:")
print("  - '<<<  -20' / '+20  >>>' buttons: Jump 20 households")
print("  - '<  Previous' / 'Next  >' buttons: Move 1 household")
print("  - Left/right arrow keys: Move 1 household")
print("  - Press 'q' to quit")
print("\nDisplay:")
print("  - All members of current household shown (compact detailed view)")
print("  - Members ordered by age (youngest to oldest)")
print("  - Households ordered by total infection count (high to low)")
print("\nLegend:")
print("  - Small gray dots: All visits")
print("  - Red dots: Fever episodes")
print("  - Open circles: Microscopy/LAMP negative")
print("  - Colored filled circles: Parasite positive (size/color = density)")
print("  - Olive edge: Gametocytes detected")
print("  - Purple filled: LAMP positive (submicroscopic)")
print("  - Text labels: Parasite density (K=thousand)")
print("=" * 80)

viewer = HouseholdViewer(df, multi_person)
plt.show()

print("\nViewer closed.")
