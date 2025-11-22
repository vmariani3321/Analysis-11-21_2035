# Prep

library(tidyverse)
library(easystats)
library(ggplot2)

    plotFont <- function(fontBase) { # Easy way to adjust font size for plots
        theme( # Add as a final ggplot object (no parentheses)
        plot.title = element_text(size = 14*fontBase),      # Title font size
        axis.title.x = element_text(size = 12*fontBase),    # X-axis title font size
        axis.title.y = element_text(size = 12*fontBase),    # Y-axis title font size
        axis.text.x = element_text(size = 10*fontBase),     # X-axis tick labels font size
        axis.text.y = element_text(size = 10*fontBase),     # Y-axis tick labels font size
        legend.text = element_text(size = 10*fontBase),     # Legend text size
        legend.title = element_text(size = 10*fontBase),    # Legend title size
        strip.text = element_text(size = 10*fontBase)
        )
    }

    
df <- readRDS("filtered_df.rds")

# Prop Tables

prop_table_def <- df %>% 
    count(definiteness, argPos) %>% 
    group_by(definiteness) %>% 
    mutate(proportion = n / sum(n))

    prop_table_def

prop_table_arg <- df %>% 
    count(argPos, definiteness) %>% 
    group_by(argPos) %>% 
    mutate(proportion = n / sum(n))

    prop_table_arg

prop_table <- df %>% 
    count(argPos, definiteness) %>% 
    mutate(proportion = n / sum(n))


    prop_table

# Model

model <- lm(data = df, surprisal ~ argPos * definiteness)
summary(model)


#Plot

plot <- ggplot(data = df, aes(x = argPos, y = surprisal, fill = definiteness)) +
    geom_boxplot(outlier.shape = NA) +
        coord_cartesian(ylim = c(0, 35)) +
    labs(
        title = "Argument Position, Definiteness, and Surprisal",
        x = "Argument Position",
        y = "Surprisal",
        fill = "Definiteness"
    ) + 
    scale_x_discrete(labels = c("sbj" = "Subject", "obj" = "Object")) +
    scale_fill_discrete(labels = c("def" = "Definite", "indef" = "Indefinite"))+
    plotFont(3) 


plot

ggsave("plot.png",
        plot = plot,
        scale = 1.5,
        units = "in",
        height = 8, width = 12,
        dpi = 600)