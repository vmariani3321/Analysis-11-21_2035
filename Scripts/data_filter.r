# Loading packages
library(tidyverse)

# Loading Files

# files <- list.files(path = "CSV",
#                     pattern = "\\.csv$",
#                     full.names = TRUE)

# df <- read_csv(files, id = "file_name")

df <- read_csv("full_csv.csv")

# Filtering

df <- df %>% 
    arrange(Sentence_ID) %>% # Removes duplicates
        group_by(Sentence_Text) %>%
        mutate(first_Sentence_ID = first(Sentence_ID)) %>%
        filter(Sentence_ID == first_Sentence_ID) %>%
        ungroup() %>%
        select(-first_Sentence_ID) %>% 
    filter(Is_NP == TRUE, # Filtering Criteria
            Is_Head_Noun == TRUE,
            Modality == "written", 
            Sent_Verb_Count == 1,
            Sent_Auxiliary_Count == 0,
            Sent_Subject_Count == 1,
            Sent_Tot_Obj_Count %in% 1,
            Sent_Dir_Object_Count == 1 ,
            Sent_Ind_Object_Count == 0,
            Sent_Sub_Conj_Count == 0,
            Sent_Coord_Conj_Count == 0, 
            Clausal_Complement_Count == 0,
            Sent_Relative_Clause_Count == 0, 
            Sent_Adv_Clause_Count == 0, 
            Sent_Prep_Phrase_Count == 0,
            Sent_Comma_Count == 0,
            !str_detect(Sentence_Text, "\\?"),
            NP_Definiteness  %in% c("definite", "indefinite"),
            NP_Argument %in% c("subject", "dir_object"),
            Sent_Transitive == TRUE,
            ) %>%
            drop_na(Phrase_Surprisal) %>% # Drops values w/out valid surprisal value
            group_by(Sentence_ID) %>% # Drops sentences without one subject and one object
                filter(n() == 2 & n_distinct(NP_Argument) == 2) %>%
                ungroup()

# Cleaning up DF leaving only what's needed for analysis

df <- df %>% 
    mutate(
        definiteness = factor(NP_Definiteness,
        levels = c("indefinite", "definite"),
        labels = c("indef", "def"))
    ) %>% 
    mutate(
        argPos = factor(
            NP_Argument,
            levels = c("dir_object", "subject"),
            labels = c("obj", "sbj")
        )
    ) %>% 
    mutate(surprisal = Phrase_Surprisal) %>% 
    select(Sentence_ID, Sentence_Text, Phrase_Token, surprisal, definiteness, argPos)

saveRDS(df, file = "filtered_df.rds")
