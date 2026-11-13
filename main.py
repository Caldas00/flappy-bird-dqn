import os
import time
import pygame
from datetime import datetime
from flappy_game import FlappyBirdEnv
from dqn_agent import Agent, device
import torch
import numpy as np

def startup_menu():
    print("=" * 50)
    print("🐦 FLAPPY RL v3 - Enhanced with Auto-Save 🐦")
    print("=" * 50)
    
    print("\n1. Create new network")
    print("2. Load existing network")
    choice = input("Select option (1/2): ").strip()
    
    load_file = None
    if choice == "2":
        print("\nAvailable models:")
        models_dir = "models"
        if os.path.exists(models_dir):
            models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
            if models:
                for i, model in enumerate(models, 1):
                    print(f"  {i}. {model}")
                try:
                    model_idx = int(input(f"Select model (1-{len(models)}): ")) - 1
                    if 0 <= model_idx < len(models):
                        load_file = os.path.join(models_dir, models[model_idx])
                    else:
                        print("Invalid selection, creating new model")
                except ValueError:
                    print("Invalid input, creating new model")
            else:
                print("  No models found, creating new network")
        else:
            print("  Models directory not found, creating new network")
    
    print("\nTraining mode:")
    print("1. Visual training (slower, see the bird learn)")
    print("2. Turbo training (fast, no rendering)")
    print("3. Hybrid mode (visual for first 1k episodes, then turbo)")
    mode = input("Select mode (1/2/3): ").strip()
    
    visual_mode = "visual" if mode == "1" else "turbo" if mode == "2" else "hybrid"
    new_model = choice == "1"
    
    return new_model, load_file, visual_mode

def print_stats(agent, episode, total_reward, episode_length, epsilon, start_time):
    """Print comprehensive training statistics"""
    stats = agent.metrics.get_stats(window=100)
    current_time = time.time()
    elapsed = current_time - start_time
    
    print(f"\n{'=' * 60}")
    print(f"Episode: {episode:,} | Reward: {total_reward:.1f} | Length: {episode_length}")
    print(f"Epsilon: {epsilon:.3f} | Training Steps: {agent.train_steps:,}")
    print(f"Elapsed Time: {elapsed/60:.1f}m | Device: {device}")
    
    if stats:
        print(f"Last 100 episodes:")
        print(f"  Mean Reward: {stats['mean_reward']:.2f} (±{stats['std_reward']:.2f})")
        print(f"  Max Reward: {stats['max_reward']:.1f}")
        print(f"  Mean Length: {stats['mean_length']:.1f}")
    
    # Show recent loss if available
    if len(agent.metrics.loss_history) > 0:
        recent_loss = np.mean(agent.metrics.loss_history[-10:])
        print(f"  Recent Loss: {recent_loss:.4f}")
    
    # Learning rate info
    current_lr = agent.optimizer.param_groups[0]['lr']
    print(f"  Learning Rate: {current_lr:.2e}")
    print(f"{'=' * 60}")

def auto_save_model(agent, episode, base_name="flappy_model_v3"):
    if episode > 0 and episode % 300 == 0:
        timestamp = datetime.now().strftime("%H%M")
        filename = f"{base_name}_ep{episode//1000}k_{timestamp}.pth"
        save_path = os.path.join("models", filename)
        
        actual_path = agent.save(save_path)
        print(f"\n🔄 AUTO-SAVED: {actual_path}")
        print(f"Episodes completed: {episode:,}")
        
        # Also save stats summary
        stats = agent.metrics.get_stats(window=1000)
        if stats:
            stats_file = save_path.replace('.pth', '_stats.txt')
            with open(stats_file, 'w') as f:
                f.write(f"Training Statistics - Episode {episode}\n")
                f.write(f"=" * 40 + "\n")
                f.write(f"Total Episodes: {episode:,}\n")
                f.write(f"Training Steps: {agent.train_steps:,}\n")
                f.write(f"Last 1000 Episodes:\n")
                f.write(f"  Mean Reward: {stats['mean_reward']:.2f}\n")
                f.write(f"  Max Reward: {stats['max_reward']:.1f}\n")
                f.write(f"  Mean Length: {stats['mean_length']:.1f}\n")
                if len(agent.metrics.loss_history) > 0:
                    f.write(f"  Recent Loss: {np.mean(agent.metrics.loss_history[-100:]):.4f}\n")
        
        return True
    return False

def main():
    # Setup
    os.makedirs("models", exist_ok=True)
    start_time = time.time()
    
    # Get user preferences
    new_model, load_file, visual_mode = startup_menu()
    
    # Initialize environment and agent
    initial_render = visual_mode in ["visual", "hybrid"]
    env = FlappyBirdEnv(render=initial_render)
    agent = Agent(state_dim=6, action_dim=2)  # Updated for 6D state
    
    # Load model if specified
    starting_episode = 0
    if load_file and os.path.isfile(load_file):
        try:
            agent.load(load_file)
            starting_episode = len(agent.metrics.episode_rewards)
            print(f"\n✅ Loaded model: {load_file}")
            print(f"Resuming from episode: {starting_episode:,}")
            epsilon = agent.get_epsilon(starting_episode)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Starting with new model...")
            epsilon = 1.0
    else:
        epsilon = 1.0
    
    # Training parameters
    max_episodes = 5000  
    print_every = 100 if visual_mode == "visual" else 1000
    stop_training = False

    # If loading an existing model and it's already trained for max_episodes, ask for more
    if starting_episode >= max_episodes:
        try:
            add_episodes = int(input(f"\nThe loaded model has already trained for {starting_episode:,} episodes.\nHow many additional episodes would you like to train? (Enter a number, or 0 to skip): "))
        except ValueError:
            add_episodes = 0
        if add_episodes > 0:
            max_episodes = starting_episode + add_episodes
            print(f"Training will continue for {add_episodes:,} more episodes (total target: {max_episodes:,}).")
        else:
            print("No additional episodes specified. Exiting training.")
            return

    last_episode = starting_episode
    
    print(f"\n🚀 Starting training on {device}")
    print(f"Target episodes: {max_episodes:,}")
    
    # Training loop
    for episode in range(starting_episode, max_episodes):
        # Switch to turbo mode after 1k episodes in hybrid mode
        if visual_mode == "hybrid" and episode == 1000:
            env = FlappyBirdEnv(render=False)
            print("\n🔄 Switching to turbo mode...")
        
        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0.0
        episode_length = 0
        episode_start_time = time.time()
        
        # Episode loop
        while not done:
            # Handle pygame events only if rendering is enabled
            if env.render_enabled:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        stop_training = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_s:  # S to stop
                            done = True
                            stop_training = True
                        elif event.key == pygame.K_SPACE:  # Space to save immediately
                            save_path = agent.save("models/manual_save")
                            print(f"\n💾 Manual save: {save_path}")
            
            # Get epsilon for current episode
            epsilon = agent.get_epsilon(episode)
            
            # Select and execute action
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            
            # Store experience and train
            agent.store(state, action, reward, next_state, done)
            agent.train_step()
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            episode_length += 1
            
            # Render if enabled
            if env.render_enabled:
                env.render()
            
        
        # Log episode metrics
        agent.metrics.log_episode(total_reward, episode_length)
        
        # Auto-save every 5k episodes
        auto_save_model(agent, episode + 1)
        
        # Print statistics
        if (episode + 1) % print_every == 0 or env.render_enabled or stop_training:
            print_stats(agent, episode + 1, total_reward, episode_length, epsilon, start_time)
        
        # Early stopping check
        if stop_training:
            print("\n🛑 Training stopped by user")
            break
        
        # Performance milestone rewards
        if total_reward > 500:
            print(f"\n🎉 Great performance! Reward: {total_reward:.1f} at episode {episode + 1}")
        elif total_reward > 200:
            print(f"\n⭐ Good run! Reward: {total_reward:.1f} at episode {episode + 1}")

        last_episode = episode
    
    # Final save and summary
    print(f"\n{'=' * 60}")
    print("🏁 TRAINING COMPLETED")
    print(f"{'=' * 60}")
    
    total_time = time.time() - start_time
    print(f"Total episodes: {last_episode + 1:,}")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Episodes per hour: {(last_episode + 1) / (total_time/3600):.0f}")
    
    # Final statistics
    final_stats = agent.metrics.get_stats(window=min(1000, len(agent.metrics.episode_rewards)))
    if final_stats:
        print(f"\nFinal Performance (last {min(1000, len(agent.metrics.episode_rewards))} episodes):")
        print(f"  Mean Reward: {final_stats['mean_reward']:.2f}")
        print(f"  Max Reward: {final_stats['max_reward']:.1f}")
        print(f"  Mean Episode Length: {final_stats['mean_length']:.1f}")
    
    # Save final model
    print(f"\n💾 Saving final model...")
    final_name = input("Enter filename for final model (or press Enter for auto): ").strip()
    if not final_name:
        final_name = f"flappy_model_v3_final_{last_episode + 1}ep"
    
    final_path = agent.save(os.path.join("models", final_name))
    print(f"✅ Final model saved: {final_path}")
    
    # Training recommendations
    print(f"\n📊 TRAINING ANALYSIS")
    print(f"{'=' * 60}")
    
    if len(agent.metrics.episode_rewards) > 100:
        recent_performance = np.mean(agent.metrics.episode_rewards[-100:])
        early_performance = np.mean(agent.metrics.episode_rewards[:100])
        improvement = recent_performance - early_performance
        
        print(f"Performance improvement: {improvement:+.1f}")
        
        if improvement > 10:
            print("✅ Strong learning progress detected!")
        elif improvement > 0:
            print("🔄 Gradual improvement - consider longer training")
        else:
            print("⚠️  Consider adjusting hyperparameters")
    
    if agent.train_steps > 10000:
        if len(agent.metrics.loss_history) > 100:
            recent_loss = np.mean(agent.metrics.loss_history[-100:])
            if recent_loss < 0.1:
                print("✅ Loss converged well")
            elif recent_loss < 1.0:
                print("🔄 Loss still decreasing - good progress")
            else:
                print("⚠️  High loss - may need learning rate adjustment")
    
    # Cleanup
    if env.render_enabled:
        pygame.quit()
    
    print(f"\n🎯 Training session complete!")
    print(f"Check the 'models' folder for saved models and statistics.")

if __name__ == "__main__":
    try:
        print("🚀 Initializing Flappy RL v3...")
        main()
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("Make sure all required packages are installed:")
        print("  pip install torch pygame numpy")
    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user (Ctrl+C)")
        try:
            pygame.quit()
        except:
            pass
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        try:
            pygame.quit()
        except:
            pass
        raise